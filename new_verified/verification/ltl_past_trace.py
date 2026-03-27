from __future__ import annotations
import re
from typing import List, Union, Any

# Formula AST: ('prop', 'p1') | ('top',) | ('bot',) | ('not', f) | ('and', [f,...]) | ('or', [f,...])
#             | ('U', f, g) | ('U-', f, g) | ('diamond_star', f)
Formula = Union[tuple, str]


def _at(trace: List[str], i: int, p: str) -> bool:
    """Proposition p holds at position i (one symbol per position)."""
    if i < 0 or i >= len(trace):
        return False
    return trace[i] == p


def _eval(trace: List[str], f: Formula, i: int) -> bool:
    """Evaluate formula f at position i (0-indexed)."""
    if not trace:
        return False
    n = len(trace)
    if isinstance(f, str):
        return _at(trace, i, f)
    tag = f[0]
    if tag == "prop":
        return _at(trace, i, f[1])
    if tag == "top":
        return True
    if tag == "bot":
        return False
    if tag == "not":
        return not _eval(trace, f[1], i)
    if tag == "and":
        return all(_eval(trace, g, i) for g in f[1])
    if tag == "or":
        return any(_eval(trace, g, i) for g in f[1])
    if tag == "U":
        # (φ U ψ) at i: ∃j≥i: ψ at j and φ at all k in [i,j)
        phi, psi = f[1], f[2]
        for j in range(i, n):
            if _eval(trace, psi, j):
                if all(_eval(trace, phi, k) for k in range(i, j)):
                    return True
        return False
    if tag == "U-":
        # (φ U− ψ) at i: ∃j<i: ψ at j and φ at all k in (j,i)
        phi, psi = f[1], f[2]
        for j in range(0, i):
            if _eval(trace, psi, j):
                if all(_eval(trace, phi, k) for k in range(j + 1, i)):
                    return True
        return False
    if tag == "diamond_star":
        # ♢⋆(φ): φ holds at the first position (index 0)
        return _eval(trace, f[1], 0)
    raise ValueError(f"Unknown formula tag: {tag}")


def evaluate(trace: List[str], formula: Formula) -> bool:
    """
    Return True iff trace satisfies formula at the first position (w |= φ).
    trace: list of symbols (one per position).
    formula: AST as produced by parse_* or build_* below.
    """
    return _eval(trace, formula, 0)


def build_prop(p: str) -> Formula:
    return ("prop", p)


def build_and(forms: List[Formula]) -> Formula:
    if len(forms) == 1:
        return forms[0]
    return ("and", forms)


def build_or(forms: List[Formula]) -> Formula:
    if len(forms) == 1:
        return forms[0]
    return ("or", forms)


def build_not(f: Formula) -> Formula:
    return ("not", f)


def build_U(phi: Formula, psi: Formula) -> Formula:
    return ("U", phi, psi)


def build_diamond_star(f: Formula) -> Formula:
    return ("diamond_star", f)


def from_internal_formula(internal: List[List[tuple]], wrap_diamond_star: bool = False) -> Formula:
    """
    Build past-LTL AST from our internal formula format (disjunction of single (a,b,c) atoms).
    Each (a,b,c) is (¬a U (b∧c)) with (b∧c) = "both b and c have appeared by this position".
    We represent "b∧c at j" as: b at j and c at j (point-in-time) is false for one-symbol-per-position;
    so we use a helper: first_bc(trace) = max(pos_b, pos_c), and (¬a U (b∧c)) holds at 0 iff
    both b and c appear and a not in trace[:first_bc]. That is exactly the semantics of our
    FormulaEvaluator. So for the past-LTL evaluator we need to encode "ψ = (b∧c) at position j"
    as "both b and c occur at or before j". We can do that with a derived operator or by
    building a formula that is true at j iff both have appeared by j: e.g. at position j,
    "b has appeared by j" is (⊤ U− b) ∨ b at 0? No. "b appears at or before j" = there exists
    k in [0,j] with trace[k]==b. That's not a simple LTL operator - it's ∃k≤j: b at k.
    In LTL we have U−: (⊤ U− b) at j means there is some position before j where b holds... no,
    (φ U− ψ) at i means ∃j<i: ψ at j and ∀k in (j,i): φ at k. So (⊤ U− b) at i means
    "b held at some position before i". So "b and c have both appeared by position i" would be
    ((⊤ U− b) ∨ b_at_0) and ((⊤ U− c) ∨ c_at_0) at i? At i=0: (⊤ U− b) is false (no j<0).
    So at 0 we need "b at 0 or (⊤ U− b) at 0" - but (⊤ U− b) at 0 is false. So at 0, "b has appeared"
    is just "b at 0". At i>0, "b has appeared by i" is (⊤ U− b) at i or b at i, or simply
    "there exists j in [0,i] with b at j". (⊤ U− b) at i means ∃j<i: b at j. So "b at some j in [0,i]"
    = b at i or (⊤ U− b) at i. So "both b and c have appeared by i" = (b_at_i or (⊤ U− b)_at_i) and
    (c_at_i or (⊤ U− c)_at_i). So we need U− in our evaluator and we need to build this.
    So (¬a U (b∧c)) in our semantics: the RHS (b∧c) at position j means "both b and c appeared by j".
    So (b∧c)_at_j = (_at(b) at j or (⊤ U− b) at j) and (_at(c) at j or (⊤ U− c) at j).
    Then (¬a U (b∧c)) = (U, not(a), that_bc_formula). So we need to build the "both appeared by j"
    subformula. Let bc_at(j) = (prop(b) or (U-, top, prop(b))) and (prop(c) or (U-, top, prop(c)))
    at j. So the formula for "both b,c by j" is: ("and", [("or", [("prop", b), ("U-", ("top",), ("prop", b))]),
    ("or", [("prop", c), ("U-", ("top",), ("prop", c))])]). Then (¬a U (b∧c)) = ("U", ("not", ("prop", a)), bc_formula).
    Let me implement that.
    """
    def bc_by_position(b: str, c: str) -> Formula:
        # Holds at j iff both b and c have appeared at or before j
        return build_and([
            build_or([build_prop(b), ("U-", ("top",), build_prop(b))]),
            build_or([build_prop(c), ("U-", ("top",), build_prop(c))]),
        ])

    def clause_to_formula(clause: List[tuple]) -> Formula:
        if not clause:
            return ("bot",)
        # No conjunctions of until terms: use only one atom per clause.
        a, b, c = clause[0]
        return build_U(build_not(build_prop(a)), bc_by_position(b, c))

    disjuncts = [clause_to_formula(conj) for conj in internal]
    inner = build_or(disjuncts)
    if wrap_diamond_star:
        inner = build_diamond_star(inner)
    return inner


def evaluate_internal_format(trace: List[str], internal: List[List[tuple]], at_first_only: bool = True) -> bool:
    """
    Evaluate our internal formula (list of disjuncts of (a,b,c) triples) on trace.
    Semantics: same as FormulaEvaluator (¬a U (b∧c) with "both appeared by j").
    If at_first_only, we evaluate at position 0 (w |= φ). Always True for this helper.
    """
    formula = from_internal_formula(internal, wrap_diamond_star=False)
    return evaluate(trace, formula)


def _black_atom_name(p: str) -> str:
    """BLACK proposition token; use raw {…} if not a simple identifier."""
    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", p):
        return p
    return "{" + p.replace("}", "\\}") + "}"


def ast_to_black(f: Formula) -> str:
    """
    Translate our AST to a BLACK input formula string (see https://www.black-sat.org/en/stable/syntax.html).
    Supports the fragment produced by from_internal_formula: prop/top/bot/not/and/or/U/U-(top only)/diamond_star.
    For (top U- psi) we emit O(Y(psi)), which matches ∃j<i: ψ at j (see internal _eval for U-).
    """
    if isinstance(f, str):
        return _black_atom_name(f)
    tag = f[0]
    if tag == "prop":
        return _black_atom_name(f[1])
    if tag == "top":
        return "True"
    if tag == "bot":
        return "False"
    if tag == "not":
        return f"(!({ast_to_black(f[1])}))"
    if tag == "and":
        parts = [ast_to_black(g) for g in f[1]]
        return "(" + " && ".join(parts) + ")"
    if tag == "or":
        parts = [ast_to_black(g) for g in f[1]]
        return "(" + " || ".join(parts) + ")"
    if tag == "U":
        return f"(({ast_to_black(f[1])}) U ({ast_to_black(f[2])}))"
    if tag == "U-":
        phi, psi = f[1], f[2]
        if phi == ("top",):
            inner = ast_to_black(psi)
            return f"(O(Y({inner})))"
        raise NotImplementedError("ast_to_black: only (top U- psi) is supported for past (use from_internal_formula)")
    if tag == "diamond_star":
        return ast_to_black(f[1])
    raise ValueError(f"ast_to_black: unknown tag {tag}")


def internal_formula_to_black(
    internal: List[List[tuple]],
    *,
    wrap_diamond_star: bool = False,
) -> str:
    """Full formula string for BLACK check (finite semantics, evaluate at state 0)."""
    return ast_to_black(from_internal_formula(internal, wrap_diamond_star=wrap_diamond_star))
