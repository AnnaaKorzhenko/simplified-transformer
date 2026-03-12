"""
LTL Formula Extractor from Simplified Transformer
Based on the PDF: Extracting LTL formulas from Transformers

Implements the extraction algorithm that converts a trained transformer
into an equivalent LTL formula.
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any
import numpy as np


class LTLExtractor:
    """
    Extracts LTL formulas from a trained SimplifiedTransformer
    """
    
    def __init__(self, model, alphabet, threshold=None):
        """
        Args:
            model: Trained SimplifiedTransformer
            alphabet: List of symbols (without omega)
            threshold: Classification threshold (uses model's threshold if None)
        """
        self.model = model
        self.alphabet = alphabet
        self.threshold = threshold if threshold is not None else model.threshold
        self.vocab_size = len(alphabet) + 1  # Including omega
        self.omega_idx = len(alphabet)
        
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
        
        # Build the LTL condition φ following LLMsREs_12 (diamond-star).
        #
        # Update rules are evaluated as global sentences (at the first time point).
        # The paper uses ♢⋆(·) to force the enclosed formula to be interpreted
        # at the beginning of the sequence even when embedded under unfolding.
        #
        # Paper shape (for given a and chosen b ∈ β_j):
        #   a ∧ ♢⋆( ∧_{b'∈β_j, b'≠b} (¬b' U b) ) ∧ ♢⋆( ∧_{b'∈β_{j'}, j'>j} ¬(⊤ U b') ) → v_{a,b}
        same_partition_parts = []
        if len(beta_j) > 1:
            for b_prime in beta_j:
                if b_prime != b:
                    same_partition_parts.append(f"(¬{b_prime} U {b})")
        else:
            # If only one element, use (⊤ U b)
            same_partition_parts.append(f"(⊤ U {b})")

        higher_partition_parts = []
        for higher_partition in beta_higher:
            for b_prime in higher_partition:
                higher_partition_parts.append(f"¬(⊤ U {b_prime})")

        same_part = " ∧ ".join(same_partition_parts) if same_partition_parts else "⊤"
        higher_part = " ∧ ".join(higher_partition_parts) if higher_partition_parts else "⊤"

        phi = f"♢⋆({same_part}) ∧ ♢⋆({higher_part})"
        
        return {
            'head': tuple(result_vec),
            'body': f"{a} ∧ {phi}",
            'symbol_a': a,
            'symbol_b': b,
            'phi': phi
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
        
        return rules, result_vectors, vector_to_symbol
    
    def unfold_formula(self, formula, rules_by_layer, layer_idx=0):
        """
        Unfold formula using the (·)° operator
        
        For layer 0: a° = a, (¬b U a)° = (¬b U a), (⊤ U b)° = (⊤ U b)
        For layer ℓ > 0: recursively unfold based on rules
        """
        if layer_idx == 0:
            return formula
        
        # This is a simplified version - full implementation would
        # recursively replace symbols based on rules
        # For now, return the formula as-is
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
        
        # Build the final formula: disjunction of unfolded rules
        if not positive_symbols:
            return "⊥"  # False
        
        # For now, return a simplified formula
        # Full implementation would properly unfold all rules
        formula_parts = []
        for rule in positive_symbols:
            formula_parts.append(rule['body'])
        
        final_formula = " ∨ ".join(formula_parts) if formula_parts else "⊥"
        
        return {
            'formula': final_formula,
            'rules_by_layer': all_rules,
            'positive_symbols': positive_symbols
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
