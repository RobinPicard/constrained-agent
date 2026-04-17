"""CLI runner for BFCL multi_turn_base evaluation with constrained_agent.

Usage:
    python -m benchmarks.bfcl.run \\
        --model Qwen/Qwen3-5B \\
        --base-url http://localhost:8000/v1 \\
        --indices 107 110 \\
        --verbose

    # Run all 200 test cases:
    python -m benchmarks.bfcl.run \\
        --model Qwen/Qwen3-5B \\
        --base-url http://localhost:8000/v1 \\
        --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker

from constrained_agent import OpenAIBackend

from .constraints import get_constraints
from .harness import run_test_case


def evaluate(result: dict, ground_truth: list[list[str]], entry: dict) -> dict:
    """Run the BFCL checker on a result."""
    return multi_turn_checker(
        multi_turn_model_result_list_decoded=result["all_model_responses"],
        multi_turn_ground_truth_list=ground_truth,
        test_entry=entry,
        test_category="multi_turn_base",
        model_name="constrained_agent",
    )


def run_single(
    backend,
    entry: dict,
    gt: dict,
    use_constraints: bool,
    inference_kwargs: dict,
    verbose: bool,
) -> dict:
    """Run a single test case and return evaluation result."""
    constraint_fns = get_constraints(entry["involved_classes"]) if use_constraints else []

    result = run_test_case(
        backend=backend,
        entry=entry,
        ground_truth=gt["ground_truth"],
        constraint_fns=constraint_fns,
        inference_kwargs=inference_kwargs,
        verbose=verbose,
    )

    checker_result = evaluate(result, gt["ground_truth"], entry)

    return {
        "test_id": entry["id"],
        "involved_classes": entry["involved_classes"],
        "constrained": use_constraints,
        "valid": checker_result.get("valid", False),
        "checker_result": checker_result,
        "turn_details": result["turn_details"],
    }


def main():
    parser = argparse.ArgumentParser(description="BFCL multi_turn_base evaluation")
    parser.add_argument(
        "--model",
        default=os.environ.get("DOTJSON_VLLM_MODEL_NAME"),
        help="Model name for the OpenAI API (default: $DOTJSON_VLLM_MODEL_NAME)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DOTJSON_VLLM_URL"),
        help="API base URL (default: $DOTJSON_VLLM_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DOTJSON_VLLM_API_KEY", "dummy"),
        help="API key (default: $DOTJSON_VLLM_API_KEY)",
    )
    parser.add_argument("--indices", type=int, nargs="+", help="Test case indices to run")
    parser.add_argument("--all", action="store_true", help="Run all 200 test cases")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per generation")
    parser.add_argument("--max-turns", type=int, default=15, help="Max agent turns per user message")
    parser.add_argument("--verbose", action="store_true", help="Print turn-by-turn debug output")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required (or set DOTJSON_VLLM_MODEL_NAME)")
    if not args.base_url:
        parser.error("--base-url is required (or set DOTJSON_VLLM_URL)")
    if not args.indices and not args.all:
        parser.error("Specify --indices or --all")

    # Load dataset
    test_entries = load_dataset_entry(
        "multi_turn_base", include_prereq=False, include_language_specific_hint=False
    )
    gt_entries = load_ground_truth_entry("multi_turn_base")
    gt_by_id = {g["id"]: g for g in gt_entries}

    indices = args.indices if args.indices else list(range(len(test_entries)))

    # Create backend
    backend = OpenAIBackend(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    inference_kwargs = {"max_tokens": args.max_tokens}

    # Run evaluations
    results = []
    constrained_pass = 0
    unconstrained_pass = 0

    for idx in indices:
        entry = test_entries[idx]
        gt = gt_by_id[entry["id"]]

        print(f"\n{'='*70}")
        print(f"Test #{idx}: {entry['id']} ({entry['involved_classes']})")
        print(f"{'='*70}")

        # Run with constraints
        r_constrained = run_single(
            backend, entry, gt,
            use_constraints=True,
            inference_kwargs=inference_kwargs,
            verbose=args.verbose,
        )
        if r_constrained["valid"]:
            constrained_pass += 1

        # Run without constraints
        r_unconstrained = run_single(
            backend, entry, gt,
            use_constraints=False,
            inference_kwargs=inference_kwargs,
            verbose=args.verbose,
        )
        if r_unconstrained["valid"]:
            unconstrained_pass += 1

        # Print comparison
        c_mark = "PASS" if r_constrained["valid"] else "FAIL"
        u_mark = "PASS" if r_unconstrained["valid"] else "FAIL"
        print(f"  Constrained:   {c_mark}")
        print(f"  Unconstrained: {u_mark}")

        if args.verbose:
            print("\n  --- Turn-by-turn (constrained) ---")
            for td in r_constrained["turn_details"]:
                match = "PASS" if td["calls"] == td["expected"] else "FAIL"
                print(f"    Turn {td['turn_idx']} [{match}]")
                print(f"      Expected: {td['expected']}")
                print(f"      Actual:   {td['calls']}")

        results.append({
            "index": idx,
            "constrained": r_constrained,
            "unconstrained": r_unconstrained,
        })

    # Summary
    total = len(indices)
    print(f"\n{'='*70}")
    print(f"SUMMARY ({total} test cases)")
    print(f"{'='*70}")
    print(f"  Constrained:   {constrained_pass}/{total} passed")
    print(f"  Unconstrained: {unconstrained_pass}/{total} passed")
    print(f"  Delta:         {constrained_pass - unconstrained_pass:+d}")

    # Detailed breakdown
    helped = []
    hurt = []
    same = []
    for r in results:
        c = r["constrained"]["valid"]
        u = r["unconstrained"]["valid"]
        if c and not u:
            helped.append(r["index"])
        elif u and not c:
            hurt.append(r["index"])
        else:
            same.append(r["index"])

    if helped:
        print(f"\n  Constraints helped ({len(helped)}): {helped}")
    if hurt:
        print(f"  Constraints hurt  ({len(hurt)}): {hurt}")
    print(f"  No difference     ({len(same)}): {len(same)} cases")

    # Save results
    if args.output:
        output = {
            "model": args.model,
            "total": total,
            "constrained_pass": constrained_pass,
            "unconstrained_pass": unconstrained_pass,
            "helped": helped,
            "hurt": hurt,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
