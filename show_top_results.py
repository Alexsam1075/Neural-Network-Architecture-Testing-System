import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_RESULTS_PATH = Path("results/test_results_v3.json")


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _pick_metric(result: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    metric_priority = (
        "accuracy",
        "score",
        "long_context_score",
        "memory_efficiency_score",
        "latency_score",
        "autoregressive_tokens_per_second",
        "inference_speed",
    )
    for key in metric_priority:
        value = _numeric(result.get(key))
        if value is not None:
            return key, value
    return None, None


def load_results(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def collect_tests(results: Dict[str, Dict[str, Dict[str, Any]]]) -> list[str]:
    tests = set()
    for architecture_results in results.values():
        tests.update(architecture_results.keys())
    return sorted(tests)


def print_top(results: Dict[str, Dict[str, Dict[str, Any]]], limit: int) -> None:
    for test_name in collect_tests(results):
        rows = []
        metric_name = None

        for architecture_name, architecture_results in results.items():
            result = architecture_results.get(test_name)
            if not isinstance(result, dict):
                continue
            metric, value = _pick_metric(result)
            if value is None:
                continue
            metric_name = metric_name or metric
            rows.append((value, architecture_name, result))

        rows.sort(key=lambda item: item[0], reverse=True)

        print("=" * 100)
        print(f"TEST: {test_name}")
        print(f"METRIC: {metric_name or 'N/A'}")
        print("=" * 100)

        if not rows:
            print("No numeric results found")
            print()
            continue

        for rank, (value, architecture_name, result) in enumerate(rows[:limit], 1):
            extras = []
            top5 = _numeric(result.get("top5_accuracy", result.get("top_5_accuracy")))
            inference_speed = _numeric(result.get("inference_speed"))
            training_time = _numeric(result.get("training_time_seconds", result.get("training_time")))

            if top5 is not None:
                extras.append(f"top5={top5:.4f}")
            if inference_speed is not None:
                extras.append(f"inf={inference_speed:.1f}/s")
            if training_time is not None:
                extras.append(f"train={training_time:.2f}s")

            suffix = f" | {'; '.join(extras)}" if extras else ""
            print(f"{rank:2d}. {architecture_name:30s} {value:.6f}{suffix}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Show top architectures by test from cached results")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_PATH), help="Path to test_results_v3.json")
    parser.add_argument("--limit", type=int, default=20, help="How many architectures to show per test")
    args = parser.parse_args()

    results = load_results(Path(args.results))
    print_top(results, args.limit)


if __name__ == "__main__":
    main()
