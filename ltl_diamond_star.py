"""
Helpers for the diamond-star (♢⋆) operator from LLMsREs_12.

The paper defines:
  ♢⋆(φ) := ( φ ∧ ¬(⊥ U− ⊤)) ∨ ⊤ U−(φ ∧ ¬(⊥ U− ⊤))

Intuition: ♢⋆(φ) is true at any time point iff φ is true at the first time point.

This repo mostly evaluates satisfaction at the beginning of the word (w |= φ iff w,1 |= φ).
Under that convention, at time point 1 we have: ♢⋆(φ) ≡ φ.
"""

from __future__ import annotations


def strip_diamond_star_at_start(formula: str) -> str:
    """
    Best-effort string rewrite for use with checkers that don't understand ♢⋆.

    This is ONLY sound when you evaluate the resulting formula at the first position
    (the repo's convention), because ♢⋆(φ) ≡ φ at time point 1.
    """
    return formula.replace("♢⋆(", "(")


def expand_diamond_star(formula: str) -> str:
    """
    Expand ♢⋆(φ) using the paper's definition, introducing U− (past-until), ⊥ and ⊤.

    Note: most external LTLf/MTL checkers will NOT support U−, so this is mainly useful
    if you have a PLTL-capable checker.
    """
    # Very small, conservative expander: only expands syntactic occurrences of "♢⋆(" by
    # turning "♢⋆(X)" into the macro with X as a parenthesized string.
    #
    # This is not a full parser (it assumes parentheses are balanced in input).
    out = []
    i = 0
    while i < len(formula):
        if formula.startswith("♢⋆(", i):
            i += 3  # len("♢⋆")
            if i >= len(formula) or formula[i] != "(":
                out.append("♢⋆")
                continue
            # parse balanced parentheses content
            depth = 0
            start = i + 1
            i += 1
            while i < len(formula):
                ch = formula[i]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    if depth == 0:
                        inner = formula[start:i]
                        first = "¬(⊥ U− ⊤)"
                        repl = f"(({inner}) ∧ {first}) ∨ (⊤ U−(({inner}) ∧ {first}))"
                        out.append(repl)
                        i += 1
                        break
                    depth -= 1
                i += 1
            else:
                # Unbalanced: emit original tail
                out.append("♢⋆(" + formula[start - 1 :])
                break
        else:
            out.append(formula[i])
            i += 1
    return "".join(out)

