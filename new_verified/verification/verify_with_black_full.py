"""
External verification using BLACK (https://www.black-sat.org/en/stable/).

Requires the `black` CLI on PATH (e.g. `brew install black-sat/black/black-sat`).

Modes:
  - full (default): same semantics as internal checker — translates internal JSON via
    ltl_past_trace.from_internal_formula + ast_to_black (future Until + past via O/Y).
  - future_plain: classical LTLf on one-symbol-per-step traces: disjunction of
    (!a U (b && c)). This does NOT match dataset labels when (b&&c) must be "both seen"
    semantics; use for comparison only.

Trace format: JSON with a finite `model` (states with true/false per proposition), as
expected by `black check --finite` (see BLACK CLI docs).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Tuple

from ltl_formulas.formula_generator import load_formula_json
from ltl_past_trace import (
    evaluate,
    from_internal_formula,
    internal_formula_to_black,
    build_U,
    build_not,
    build_prop,
    build_and,
    build_or,
)


def find_black() -> str | None:
    return shutil.which("black")


def trace_to_black_json(sequence: List[str], alphabet: List[str]) -> Dict[str, Any]:
    """Finite trace as BLACK JSON (no 'loop' field; size matches state count)."""
    states = []
    for sym in sequence:
        st = {p: ("true" if p == sym else "false") for p in alphabet}
        states.append(st)
    return {
        "model": {
            "size": len(states),
            "states": states,
        }
    }


def formula_future_plain_black(formula: List[List[Tuple[str, str, str]]]) -> str:
    """Standard (!a U (b && c)) disjuncts — not 'both seen' semantics."""
    parts = []
    for clause in formula:
        if not clause:
            continue
        a, b, c = clause[0]
        parts.append(f"((!{a}) U ({b} && {c}))")
    if not parts:
        return "false"
    if len(parts) == 1:
        return parts[0]
    return "(" + " || ".join(parts) + ")"


def future_plain_ast(formula: List[List[Tuple[str, str, str]]]):
    """Classical (!a U (b && c)) per disjunct; same string as formula_future_plain_black."""
    disj: List = []
    for clause in formula:
        if not clause:
            continue
        a, b, c = clause[0]
        disj.append(
            build_U(
                build_not(build_prop(a)),
                build_and([build_prop(b), build_prop(c)]),
            )
        )
    if not disj:
        return ("bot",)
    return build_or(disj)


def black_check(
    formula_str: str,
    trace_obj: Dict[str, Any],
    *,
    initial_state: int = 0,
    black_bin: str = "black",
) -> Tuple[bool, str]:
    """
    Run `black check --finite -i <initial> -f <formula> -t <trace.json>`.
    Returns (satisfied, combined stdout/stderr for logging).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pltl", delete=False, encoding="utf-8"
    ) as ff:
        ff.write(formula_str)
        formula_path = ff.name
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tf:
        json.dump(trace_obj, tf)
        trace_path = tf.name
    try:
        proc = subprocess.run(
            [
                black_bin,
                "check",
                "--finite",
                "-i",
                str(initial_state),
                "-t",
                trace_path,
                formula_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        ok = "TRUE" in (proc.stdout or "")
        return ok, out.strip()
    finally:
        try:
            os.unlink(formula_path)
        except OSError:
            pass
        try:
            os.unlink(trace_path)
        except OSError:
            pass


def verify_full_dataset(
    formula_file: str,
    dataset_file: str,
    *,
    mode: str = "full",
    wrap_diamond_star: bool = False,
    black_bin: str | None = None,
) -> dict:
    formula, metadata = load_formula_json(formula_file)
    alphabet = list(metadata.get("alphabet", []))
    black_path = black_bin or find_black()

    if mode == "full":
        formula_str = internal_formula_to_black(
            formula, wrap_diamond_star=wrap_diamond_star
        )
        py_ast = from_internal_formula(formula, wrap_diamond_star=wrap_diamond_star)
    elif mode == "future_plain":
        formula_str = formula_future_plain_black(formula)
        py_ast = future_plain_ast(formula)
    else:
        return {"error": f"Unknown mode: {mode}"}

    dataset: List[Tuple[List[str], int]] = []
    with open(dataset_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = [s.strip() for s in row["sequence"].split(",")]
            dataset.append((seq, int(row["label"])))

    use_cli = black_path is not None
    method_note = (
        f"BLACK CLI subprocess ({mode})"
        if use_cli
        else "Python evaluation (same AST as internal_formula_to_black; install `black` for external check)"
    )

    correct = 0
    errors: List[dict] = []
    for seq, expected_label in dataset:
        if use_cli:
            trace_obj = trace_to_black_json(seq, alphabet)
            ok, _log = black_check(
                formula_str, trace_obj, initial_state=0, black_bin=black_path
            )
        else:
            ok = evaluate(seq, py_ast)
        actual = 1 if ok else 0
        if actual == expected_label:
            correct += 1
        elif len(errors) < 10:
            errors.append(
                {
                    "sequence": ",".join(seq),
                    "expected": expected_label,
                    "actual": actual,
                }
            )

    n = len(dataset)
    return {
        "total_checked": n,
        "correct": correct,
        "incorrect": n - correct,
        "accuracy": (100.0 * correct / n) if n else 0.0,
        "errors": errors,
        "verification_method": method_note,
        "black_cli_used": use_cli,
        "black_formula": formula_str[:2000] + ("..." if len(formula_str) > 2000 else ""),
        "wrap_diamond_star": wrap_diamond_star,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Verify dataset with BLACK (external LTLf + past)")
    p.add_argument(
        "--formula",
        "-f",
        default="generated_diamond_star/formulas/formula_1.json",
        help="Formula JSON path",
    )
    p.add_argument(
        "--dataset",
        "-d",
        default="generated_diamond_star/datasets/dataset_1.csv",
        help="Dataset CSV path",
    )
    p.add_argument(
        "--mode",
        choices=("full", "future_plain"),
        default="full",
        help="full = internal+past AST to BLACK; future_plain = classical (!a U (b&&c))",
    )
    p.add_argument(
        "--diamond-star",
        action="store_true",
        help="Wrap formula in diamond-star (same as internal past-LTL wrapper)",
    )
    p.add_argument("--black-bin", default=None, help="Path to black executable")
    p.add_argument("-o", "--output", default="black_verification_results.json")
    args = p.parse_args()

    if not os.path.exists(args.formula) or not os.path.exists(args.dataset):
        print("Missing formula or dataset file.")
        return

    res = verify_full_dataset(
        args.formula,
        args.dataset,
        mode=args.mode,
        wrap_diamond_star=args.diamond_star,
        black_bin=args.black_bin,
    )
    if "error" in res:
        print(res["error"])
        return

    print(json.dumps(res, indent=2))
    with open(args.output, "w", encoding="utf-8") as out:
        json.dump(res, out, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
