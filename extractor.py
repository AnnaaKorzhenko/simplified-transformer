"""
LTL Formula Extractor from Simplified Transformer.
Based on: Extracting LTL formulas from Transformers (LLMsREs_15.pdf).

Implements the rule-extraction algorithm that converts a trained transformer
into an equivalent LTL formula. Rule shape (LLMsREs_15 §3):
  a ∧ ♢⋆(∧_{b'∈β_j, b'≠b} (¬b' U b)) ∧ ♢⋆(∧_{b'∈β_{j'}, j'>j} ¬(⊤ U b')) → v_{a,b}
with (⊤ U b) when β_j is a singleton. ♢⋆φ means φ holds at the first position (§2).
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any, Optional
import numpy as np


class LTLExtractor:
    """
    Extracts LTL formulas from a trained SimplifiedTransformer.
    Rules use begin-anchored diamond-star (LLMsREs section 2--3): align readout with global satisfaction.
    """
    
    def __init__(self, model, alphabet, threshold=None, unfold_node_budget: int = 20000):
        """
        Args:
            model: Trained SimplifiedTransformer
            alphabet: List of symbols (without omega)
            threshold: Classification threshold (uses model's threshold if None)
            unfold_node_budget: max AST nodes visited during unfolding
        """
        self.model = model
        self.alphabet = alphabet
        self.threshold = threshold if threshold is not None else model.threshold
        self.vocab_size = len(alphabet) + 1  # Including omega
        self.omega_idx = len(alphabet)
        self.base_symbols = set(alphabet) | {"ω"}
        self._unfold_node_budget = int(unfold_node_budget)
        self._unfold_node_count = 0
        self._unfold_truncated = False

    # --------------------------- AST helpers ---------------------------
    # AST node shapes:
    #   ('sym', s), ('top',), ('bot',), ('not', x), ('u', l, r),
    #   ('diamond_star', x), ('and', [x1,..]), ('or', [x1,..])
    def _sym(self, s: str):
        return ("sym", s)

    def _top(self):
        return ("top",)

    def _bot(self):
        return ("bot",)

    def _not(self, x):
        return ("not", x)

    def _u(self, left, right):
        return ("u", left, right)

    def _ds(self, x):
        return ("diamond_star", x)

    def _and(self, xs: List[Any]):
        return ("and", xs)

    def _or(self, xs: List[Any]):
        return ("or", xs)

    def _node_repr(self, node: Any) -> str:
        return repr(node)

    def _guard_nodes(self) -> bool:
        self._unfold_node_count += 1
        if self._unfold_node_count > self._unfold_node_budget:
            self._unfold_truncated = True
            return False
        return True

    def _simplify(self, node: Any) -> Any:
        """Lightweight simplification to limit formula blow-up."""
        kind = node[0]
        if kind in ("sym", "top", "bot"):
            return node
        if kind == "not":
            x = self._simplify(node[1])
            if x[0] == "top":
                return self._bot()
            if x[0] == "bot":
                return self._top()
            if x[0] == "not":
                return self._simplify(x[1])
            return self._not(x)
        if kind == "u":
            l = self._simplify(node[1])
            r = self._simplify(node[2])
            # (⊤ U x) remains useful as "seen x"; keep it.
            if r[0] == "top":
                return self._top()
            if l[0] == "bot" and r[0] == "bot":
                return self._bot()
            return self._u(l, r)
        if kind == "diamond_star":
            x = self._simplify(node[1])
            return self._ds(x)
        if kind in ("and", "or"):
            parts = []
            seen = set()
            for p in node[1]:
                q = self._simplify(p)
                if q[0] == kind:
                    for qq in q[1]:
                        key = self._node_repr(qq)
                        if key not in seen:
                            parts.append(qq)
                            seen.add(key)
                else:
                    key = self._node_repr(q)
                    if key not in seen:
                        parts.append(q)
                        seen.add(key)

            if kind == "and":
                if any(p[0] == "bot" for p in parts):
                    return self._bot()
                parts = [p for p in parts if p[0] != "top"]
            else:
                if any(p[0] == "top" for p in parts):
                    return self._top()
                parts = [p for p in parts if p[0] != "bot"]

            if len(parts) == 0:
                return self._top() if kind == "and" else self._bot()
            if len(parts) == 1:
                return parts[0]
            return (kind, parts)

        return node

    def _ast_to_string(self, node: Any) -> str:
        kind = node[0]
        if kind == "sym":
            return str(node[1])
        if kind == "top":
            return "⊤"
        if kind == "bot":
            return "⊥"
        if kind == "not":
            return f"¬({self._ast_to_string(node[1])})"
        if kind == "u":
            return f"({self._ast_to_string(node[1])} U {self._ast_to_string(node[2])})"
        if kind == "diamond_star":
            return f"♢⋆({self._ast_to_string(node[1])})"
        if kind == "and":
            return " ∧ ".join(f"({self._ast_to_string(x)})" for x in node[1])
        if kind == "or":
            return " ∨ ".join(f"({self._ast_to_string(x)})" for x in node[1])
        return str(node)

    def _symbol_layer(self, s: str) -> int:
        """Return Σ-layer index for symbol: base alphabet -> 0, σ_k_* -> k."""
        if s in self.base_symbols:
            return 0
        if s.startswith("σ_"):
            parts = s.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
        return 0

    def _build_rule_body_ast(self, a: str, b: str, beta_j: Set[str], beta_higher: List[Set[str]]):
        # same partition: if singleton use (⊤ U b), else ∧_{b'!=b}(¬b' U b)
        same_parts = []
        if len(beta_j) > 1:
            for b_prime in sorted(beta_j):
                if b_prime != b:
                    same_parts.append(self._u(self._not(self._sym(b_prime)), self._sym(b)))
        else:
            same_parts.append(self._u(self._top(), self._sym(b)))

        higher_parts = []
        for higher in beta_higher:
            for b_prime in sorted(higher):
                higher_parts.append(self._not(self._u(self._top(), self._sym(b_prime))))

        same_part = self._and(same_parts) if same_parts else self._top()
        higher_part = self._and(higher_parts) if higher_parts else self._top()
        return self._and([self._sym(a), self._ds(same_part), self._ds(higher_part)])

    def _build_rule_index(self, all_rules: List[Dict[str, Any]]) -> Dict[Tuple[int, str], Dict[str, Any]]:
        """
        Build mapping (Σ-layer, head_symbol) -> rule.
        Rules extracted at layer_idx produce Σ_{layer_idx+1} symbols.
        """
        idx = {}
        for layer_pack in all_rules:
            sigma_layer = layer_pack["layer"] + 1
            for r in layer_pack["rules"]:
                hs = r.get("head_symbol")
                if hs is None:
                    continue
                idx[(sigma_layer, hs)] = r
        return idx

    def _unfold_ast(self, node: Any, layer_prev: int, rule_index: Dict[Tuple[int, str], Dict[str, Any]], memo: Dict[Tuple[str, int], Any]) -> Any:
        """Unfold symbols in AST that belong to Σ_{layer_prev}."""
        if not self._guard_nodes():
            return self._top()
        kind = node[0]
        if kind == "sym":
            s = node[1]
            if self._symbol_layer(s) != layer_prev:
                return node
            return self._unfold_symbol(s, layer_prev, rule_index, memo)
        if kind in ("top", "bot"):
            return node
        if kind == "not":
            return self._not(self._unfold_ast(node[1], layer_prev, rule_index, memo))
        if kind == "u":
            return self._u(
                self._unfold_ast(node[1], layer_prev, rule_index, memo),
                self._unfold_ast(node[2], layer_prev, rule_index, memo),
            )
        if kind == "diamond_star":
            return self._ds(self._unfold_ast(node[1], layer_prev, rule_index, memo))
        if kind in ("and", "or"):
            return (kind, [self._unfold_ast(x, layer_prev, rule_index, memo) for x in node[1]])
        return node

    def _unfold_symbol(self, sym: str, sigma_layer: int, rule_index: Dict[Tuple[int, str], Dict[str, Any]], memo: Dict[Tuple[str, int], Any]) -> Any:
        """
        Compute sym^{◦,sigma_layer}.
        Base: sigma_layer == 0 -> symbol itself.
        Recurrence: a^{◦,ℓ} = body(r)^{◦,ℓ-1} where head(r)=a.
        """
        key = (sym, sigma_layer)
        if key in memo:
            return memo[key]

        if sigma_layer == 0 or sym in self.base_symbols:
            memo[key] = self._sym(sym)
            return memo[key]

        rule = rule_index.get((sigma_layer, sym))
        if rule is None:
            # Unknown symbol -> keep symbolic (safe fallback).
            memo[key] = self._sym(sym)
            return memo[key]

        body_ast = rule["body_ast"]  # symbols from Σ_{sigma_layer-1}
        unfolded = self._unfold_ast(body_ast, sigma_layer - 1, rule_index, memo)
        memo[key] = self._simplify(unfolded)
        return memo[key]
        
    def get_all_vectors_in_layer(self, layer_idx):
        """
        Get all unique vectors that can appear in layer layer_idx
        This is a simplified version - in practice, we'd need to enumerate
        all possible combinations, but for efficiency we sample or use
        the vectors we've seen.
        """
        vectors = set()
        
        # Get vectors from encoding matrix for layer 0
        if layer_idx == 0:
            for i in range(self.vocab_size):
                vec = self.model.W_enc[i].detach().cpu().numpy()
                vectors.add(tuple(vec))
        else:
            # For deeper layers, we need to compute all possible vectors
            # This is computationally expensive, so we use a sampling approach
            # or compute for a representative set of inputs
            pass
            
        return vectors
    
    def compute_attention_scores(self, layer_idx, vec_a, vec_b):
        """
        Compute attention score α_{a,b} = x_{ℓ-1,a} Q^ℓ (K^ℓ)^T x_{ℓ-1,b}^T
        
        Args:
            layer_idx: Layer index (0-indexed, so layer 1 is idx 0)
            vec_a: Vector for symbol a
            vec_b: Vector for symbol b
            
        Returns:
            Attention score
        """
        Q = self.model.Q_layers[layer_idx]
        K = self.model.K_layers[layer_idx]
        
        vec_a_tensor = torch.tensor(vec_a, dtype=torch.float32).unsqueeze(0)
        vec_b_tensor = torch.tensor(vec_b, dtype=torch.float32).unsqueeze(0)
        
        # Compute Q^ℓ and K^ℓ
        Q_a = Q(vec_a_tensor)  # [1, d_prime]
        K_b = K(vec_b_tensor)  # [1, d_prime]
        
        # Compute attention score: Q_a @ K_b^T
        score = torch.matmul(Q_a, K_b.transpose(0, 1)).item()
        
        return score
    
    def partition_symbols_by_attention(self, layer_idx, symbol_vectors, query_symbol_vec):
        """
        Partition symbols based on attention scores for a given query symbol
        
        Returns:
            List of partitions, each partition is a set of symbols with same attention score
            Partitions are ordered by decreasing attention score
        """
        attention_scores = {}
        
        for symbol, vec in symbol_vectors.items():
            score = self.compute_attention_scores(layer_idx, query_symbol_vec, vec)
            attention_scores[symbol] = score
        
        # Group by score value
        score_to_symbols = defaultdict(set)
        for symbol, score in attention_scores.items():
            score_to_symbols[score].add(symbol)
        
        # Sort partitions by score (descending)
        sorted_scores = sorted(score_to_symbols.keys(), reverse=True)
        partitions = [score_to_symbols[score] for score in sorted_scores]
        
        return partitions, attention_scores
    
    def create_rule(self, a, b, beta_j, beta_higher, layer_idx, symbol_vectors_a, symbol_vectors_b):
        """
        Create a rule: a ∧ φ → a'
        where φ involves LTL operators based on attention patterns
        """
        # Compute the resulting vector a'
        vec_a = symbol_vectors_a[a]
        vec_b = symbol_vectors_b[b]
        
        V = self.model.V_layers[layer_idx]
        O = self.model.O_layers[layer_idx]
        
        vec_a_tensor = torch.tensor(vec_a, dtype=torch.float32).unsqueeze(0)
        vec_b_tensor = torch.tensor(vec_b, dtype=torch.float32).unsqueeze(0)
        
        # Compute σ(x_a V^ℓ O^ℓ) + x_b
        attended = V(vec_a_tensor)  # [1, d_prime]
        output = O(attended)  # [1, d_model]
        activated = self.model.activation(output)  # [1, d_model]
        result_vec = (activated + vec_b_tensor).squeeze(0).detach().cpu().numpy()
        
        body_ast = self._build_rule_body_ast(a, b, beta_j, beta_higher)
        phi = self._ast_to_string(self._simplify(self._and(body_ast[1][1:])))
        
        return {
            'head': tuple(result_vec),
            'body': self._ast_to_string(self._simplify(body_ast)),
            'body_ast': body_ast,
            'symbol_a': a,
            'symbol_b': b,
            'phi': phi,
        }
    
    def extract_layer_rules(self, layer_idx, symbol_vectors_prev):
        """
        Extract rules for layer layer_idx
        
        Returns:
            List of rules and mapping from result vectors to symbols
        """
        rules = []
        result_vectors = {}
        vector_to_symbol = {}
        
        # For each symbol a in previous layer
        for a, vec_a in symbol_vectors_prev.items():
            # Partition symbols by attention scores
            partitions, attention_scores = self.partition_symbols_by_attention(
                layer_idx, symbol_vectors_prev, vec_a
            )
            
            # For each partition
            for j, beta_j in enumerate(partitions):
                beta_higher = partitions[j+1:] if j+1 < len(partitions) else []
                
                # For each symbol b in this partition
                for b in beta_j:
                    rule = self.create_rule(
                        a, b, beta_j, beta_higher, layer_idx,
                        symbol_vectors_prev, symbol_vectors_prev
                    )
                    rules.append(rule)
                    
                    # Map result vector to a symbol
                    if rule['head'] not in vector_to_symbol:
                        symbol_name = f"σ_{layer_idx+1}_{len(vector_to_symbol)}"
                        vector_to_symbol[rule['head']] = symbol_name
                        result_vectors[symbol_name] = rule['head']
                    rule['head_symbol'] = vector_to_symbol[rule['head']]
        
        return rules, result_vectors, vector_to_symbol
    
    def unfold_formula(self, formula, rules_by_layer, layer_idx=0):
        """
        Unfold formula using the (·)° operator
        
        For layer 0: a° = a, (¬b U a)° = (¬b U a), (⊤ U b)° = (⊤ U b)
        For layer ℓ > 0: recursively unfold based on rules
        """
        # Backward-compatible helper: if `formula` is a final-layer symbol name,
        # unfold it to Σ0 using memoized DAG unfolding.
        if isinstance(formula, str):
            sigma_layer = self._symbol_layer(formula)
            idx = self._build_rule_index(rules_by_layer)
            memo: Dict[Tuple[str, int], Any] = {}
            self._unfold_node_count = 0
            self._unfold_truncated = False
            ast = self._unfold_symbol(formula, sigma_layer, idx, memo)
            ast = self._simplify(ast)
            return self._ast_to_string(ast)
        return formula
    
    def extract_formula(self):
        """
        Extract the final LTL formula from the transformer
        
        Returns:
            LTL formula as a string
        """
        # Get initial symbol vectors (layer 0)
        symbol_vectors = {}
        for i, symbol in enumerate(self.alphabet):
            vec = self.model.W_enc[i].detach().cpu().numpy()
            symbol_vectors[symbol] = tuple(vec)
        
        # Add omega
        omega_vec = self.model.W_enc[self.omega_idx].detach().cpu().numpy()
        symbol_vectors['ω'] = tuple(omega_vec)
        
        # Extract rules for each layer
        all_rules = []
        current_symbols = symbol_vectors.copy()
        
        for layer_idx in range(self.model.num_layers):
            rules, new_symbols, vector_to_symbol = self.extract_layer_rules(
                layer_idx, current_symbols
            )
            all_rules.append({
                'layer': layer_idx,
                'rules': rules,
                'symbols': new_symbols,
                'vector_to_symbol': vector_to_symbol
            })
            current_symbols = new_symbols
        
        # Find symbols in final layer that lead to True (value > threshold)
        final_layer_idx = self.model.num_layers - 1
        final_rules = all_rules[final_layer_idx]['rules']
        
        positive_symbols = []
        for rule in final_rules:
            # Check if the result vector has first element > threshold
            result_vec = np.array(rule['head'])
            if result_vec[0] > self.threshold:
                positive_symbols.append(rule)
        
        # Build final formula by unfolding positive final-layer symbols.
        if not positive_symbols:
            return "⊥"  # False

        rule_index = self._build_rule_index(all_rules)
        memo: Dict[Tuple[str, int], Any] = {}
        self._unfold_node_count = 0
        self._unfold_truncated = False

        unfolded_parts = []
        for rule in positive_symbols:
            hs = rule.get("head_symbol")
            if hs is None:
                continue
            sigma_layer = self._symbol_layer(hs)
            unfolded_parts.append(self._unfold_symbol(hs, sigma_layer, rule_index, memo))

        if not unfolded_parts:
            final_ast = self._bot()
        else:
            final_ast = self._simplify(self._or(unfolded_parts))
        final_formula = self._ast_to_string(final_ast)

        return {
            'formula': final_formula,
            'formula_ast': final_ast,
            'rules_by_layer': all_rules,
            'positive_symbols': positive_symbols
            ,
            'unfold_stats': {
                'memo_entries': len(memo),
                'node_budget': self._unfold_node_budget,
                'node_visits': self._unfold_node_count,
                'truncated': self._unfold_truncated
            }
        }
    
    def extract_formula_simplified(self, max_depth=3):
        """
        Simplified extraction that focuses on the final layer
        and creates a readable LTL formula
        """
        # Get all initial symbols
        initial_symbols = {}
        for i, symbol in enumerate(self.alphabet):
            vec = self.model.W_enc[i].detach().cpu().numpy()
            initial_symbols[symbol] = vec
        
        # Simulate forward pass for omega position
        # We focus on what happens at the last position (omega)
        omega_vec = self.model.W_enc[self.omega_idx].detach().cpu().numpy()
        
        # Track what the model learns to recognize
        # This is a heuristic approach for demonstration
        X = torch.tensor(omega_vec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Process through layers
        for layer_idx in range(self.model.num_layers):
            Q = self.model.Q_layers[layer_idx]
            K = self.model.K_layers[layer_idx]
            V = self.model.V_layers[layer_idx]
            O = self.model.O_layers[layer_idx]
            
            # Compute attention (simplified - in practice need full sequence)
            Q_vec = Q(X.squeeze(0))
            K_vec = K(X.squeeze(0))
            scores = torch.matmul(Q_vec, K_vec.transpose(0, 1))
            
            # Get attended vector
            attended = V(X.squeeze(0))
            output = O(attended)
            activated = self.model.activation(output)
            X = activated + X
        
        # Check final value
        final_value = X[0, 0, 0].item()
        
        # Build a simplified formula representation
        # In practice, this would be more complex
        if final_value > self.threshold:
            # Model accepts - create a formula that represents the pattern
            # This is a placeholder - real extraction is more complex
            formula = "∃ pattern that leads to acceptance"
        else:
            formula = "⊥"
        
        return formula
