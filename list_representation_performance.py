


"""
Find the best-performing representation for each dataset and plane.
Can be used to select what representations to use when training PaSNet based on baseline performance.
Use the validation performance to guide the selection in order to prevent test-set leakage unless all combinations are evaluated for PaSNet (but then you won't need this script for the selection part).

The script scans all files ending in "summary.json" in a folder, reads:

    args.dimensions
    meanValAccuracy
    meanAccuracy

Plane mapping:
    dimensions starting with "xt"   -> plane "xt"
    dimensions starting with "ty"   -> plane "ty"
    dimensions starting with "cstr" -> plane "xy"

For every dataset and plane, the script reports:
    1. The representation with the best validation accuracy.
    2. The representation with the best test accuracy.

Usage:
    python find_best_representations.py /path/to/results

To scan subfolders recursively:
    python find_best_representations.py /path/to/results --recursive

To also save the results:
    python find_best_representations.py /path/to/results --csv results.csv
    python find_best_representations.py /path/to/results --json results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DATASETS = [
    "cifar10_dvs",
    "ncaltech101",
    "ncars",
    "dvs_gesture",
    "asl_dvs",
    "sl_animals_dvs",
    "daily_action_dvs",
    "thu_eact_50_chl",
    "daily_dvs_200",
    "dvs_lip",
]


@dataclass(frozen=True)
class Result:
    dataset: str
    plane: str
    representation: str
    validation_accuracy: float
    test_accuracy: float
    file: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report the best validation and test representation for each "
            "dataset and plane."
        )
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing the result JSON files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for summary JSON files in subfolders as well.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        dest="csv_output",
        help="Optional path for a CSV output file.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        dest="json_output",
        help="Optional path for a JSON output file.",
    )
    return parser.parse_args()


def identify_dataset(filename: str) -> str | None:
    """
    Identify a dataset from the beginning of a filename.

    An underscore, semicolon, or end of string must follow the dataset name.
    This prevents accidental partial matches.
    """
    for dataset in sorted(DATASETS, key=len, reverse=True):
        if filename == dataset:
            return dataset

        for separator in ("_", ";"):
            if filename.startswith(dataset + separator):
                return dataset

    return None


def identify_plane(representation: str) -> str | None:
    """
    Convert the representation/dimensions prefix to a plane name.
    """
    representation = representation.strip().lower()

    if representation.startswith("xt"):
        return "xt"
    if representation.startswith("ty"):
        return "ty"
    if representation.startswith("cstr"):
        return "xy"

    return None


def to_finite_float(value: Any, key: str, path: Path) -> float:
    """
    Convert a JSON value to a finite float.
    """
    if isinstance(value, bool):
        raise ValueError(f"{key!r} is a boolean, not a numeric score")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key!r} is not numeric: {value!r}") from exc

    if not math.isfinite(number):
        raise ValueError(f"{key!r} is not finite: {value!r}")

    return number


def read_result(path: Path) -> Result:
    """
    Read and validate one summary JSON file.
    """
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    except OSError as exc:
        raise ValueError(f"could not read file: {exc}") from exc

    dataset = identify_dataset(path.name)
    if dataset is None:
        raise ValueError("filename does not start with a recognized dataset")

    args = data.get("args")
    if not isinstance(args, dict):
        raise ValueError("missing or invalid 'args' object")

    representation = args.get("dimensions")
    if not isinstance(representation, str) or not representation.strip():
        raise ValueError("missing or invalid 'args.dimensions' string")

    representation = representation.strip()
    plane = identify_plane(representation)
    if plane is None:
        raise ValueError(
            "args.dimensions does not start with 'xt', 'ty', or 'cstr'"
        )

    validation_accuracy = to_finite_float(
        data.get("meanValAccuracy"),
        "meanValAccuracy",
        path,
    )
    test_accuracy = to_finite_float(
        data.get("meanAccuracy"),
        "meanAccuracy",
        path,
    )

    return Result(
        dataset=dataset,
        plane=plane,
        representation=representation,
        validation_accuracy=validation_accuracy,
        test_accuracy=test_accuracy,
        file=str(path),
    )


def find_summary_files(folder: Path, recursive: bool) -> Iterable[Path]:
    """
    Yield files whose names end with 'summary.json'.
    """
    pattern = "**/*summary.json" if recursive else "*summary.json"

    for path in sorted(folder.glob(pattern)):
        if path.is_file():
            yield path


def load_results(
    folder: Path,
    recursive: bool,
) -> tuple[list[Result], list[tuple[Path, str]]]:
    results: list[Result] = []
    errors: list[tuple[Path, str]] = []

    for path in find_summary_files(folder, recursive):
        try:
            results.append(read_result(path))
        except ValueError as exc:
            errors.append((path, str(exc)))

    return results, errors


def get_best_results(
    results: Iterable[Result],
) -> dict[tuple[str, str], dict[str, Result]]:
    """
    Return the best validation and test result for every dataset-plane pair.

    On an exact score tie, the lexicographically smaller representation is
    selected to keep the output deterministic.
    """
    best: dict[tuple[str, str], dict[str, Result]] = {}

    for result in results:
        key = (result.dataset, result.plane)
        group = best.setdefault(key, {})

        current_validation = group.get("validation")
        if (
            current_validation is None
            or result.validation_accuracy
            > current_validation.validation_accuracy
            or (
                result.validation_accuracy
                == current_validation.validation_accuracy
                and result.representation < current_validation.representation
            )
        ):
            group["validation"] = result

        current_test = group.get("test")
        if (
            current_test is None
            or result.test_accuracy > current_test.test_accuracy
            or (
                result.test_accuracy == current_test.test_accuracy
                and result.representation < current_test.representation
            )
        ):
            group["test"] = result

    return best


def format_score(score: float) -> str:
    """
    Format a score without assuming whether it is stored as 0-1 or 0-100.
    """
    return f"{score:.6f}"


def sort_plane_results(results: Iterable[Result]) -> list[Result]:
    """
    Sort all results for a dataset-plane combination.

    Results are ordered primarily by validation accuracy, then by test
    accuracy, both descending.
    """
    return sorted(
        results,
        key=lambda result: (
            -result.validation_accuracy,
            -result.test_accuracy,
            result.representation,
            result.file,
        ),
    )

def print_report(
    best: dict[tuple[str, str], dict[str, Result]],
    results: Iterable[Result],
) -> None:
    dataset_order = {dataset: index for index, dataset in enumerate(DATASETS)}
    plane_order = {"xy": 0, "xt": 1, "ty": 2}

    # Keep all results grouped by dataset and plane so they can be displayed
    # when the validation and test winners differ.
    grouped_results: dict[tuple[str, str], list[Result]] = {}

    for result in results:
        key = (result.dataset, result.plane)
        grouped_results.setdefault(key, []).append(result)

    sorted_keys = sorted(
        best,
        key=lambda key: (
            dataset_order.get(key[0], len(DATASETS)),
            plane_order.get(key[1], 99),
        ),
    )

    if not sorted_keys:
        print("No valid result files were found.")
        return

    for dataset, plane in sorted_keys:
        group = best[(dataset, plane)]
        validation = group["validation"]
        test = group["test"]

        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset}")
        print(f"Plane:   {plane}")
        print(f"{'=' * 80}")

        print(
            "Best validation:"
            f" {format_score(validation.validation_accuracy)}"
        )
        print(f"  Representation: {validation.representation}")
        print(f"  Test score:     {format_score(validation.test_accuracy)}")
        print(f"  File:           {validation.file}")

        print(f"\nBest test:       {format_score(test.test_accuracy)}")
        print(f"  Representation: {test.representation}")
        print(
            "  Validation score:"
            f" {format_score(test.validation_accuracy)}"
        )
        print(f"  File:            {test.file}")

        if validation.representation != test.representation:
            print(
                "\nThe best validation and test representations differ."
                "\nAll results for this dataset and plane:"
            )

            plane_results = sort_plane_results(
                grouped_results[(dataset, plane)]
            )

            for index, result in enumerate(plane_results, start=1):
                labels: list[str] = []

                if result is validation:
                    labels.append("BEST VALIDATION")

                if result is test:
                    labels.append("BEST TEST")

                label_text = (
                    f" [{', '.join(labels)}]"
                    if labels
                    else ""
                )

                print(f"\n  {index}. {result.representation}{label_text}")
                print(
                    "     Validation:"
                    f" {format_score(result.validation_accuracy)}"
                )
                print(
                    "     Test:      "
                    f" {format_score(result.test_accuracy)}"
                )
                print(f"     File:       {result.file}")
                

def create_output_rows(
    best: dict[tuple[str, str], dict[str, Result]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for dataset, plane in sorted(best):
        validation = best[(dataset, plane)]["validation"]
        test = best[(dataset, plane)]["test"]

        rows.append(
            {
                "dataset": dataset,
                "plane": plane,
                "best_validation_representation": validation.representation,
                "best_validation_accuracy": validation.validation_accuracy,
                "best_validation_test_accuracy": validation.test_accuracy,
                "best_validation_file": validation.file,
                "best_test_representation": test.representation,
                "best_test_accuracy": test.test_accuracy,
                "best_test_validation_accuracy": test.validation_accuracy,
                "best_test_file": test.file,
            }
        )

    return rows


def save_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "plane",
        "best_validation_representation",
        "best_validation_accuracy",
        "best_validation_test_accuracy",
        "best_validation_file",
        "best_test_representation",
        "best_test_accuracy",
        "best_test_validation_accuracy",
        "best_test_file",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(
    best: dict[tuple[str, str], dict[str, Result]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output: dict[str, dict[str, dict[str, Any]]] = {}

    for (dataset, plane), group in sorted(best.items()):
        output.setdefault(dataset, {})[plane] = {
            "best_validation": asdict(group["validation"]),
            "best_test": asdict(group["test"]),
        }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)


def main() -> int:
    arguments = parse_arguments()
    folder = arguments.folder.expanduser().resolve()

    if not folder.exists():
        print(f"Error: folder does not exist: {folder}", file=sys.stderr)
        return 1

    if not folder.is_dir():
        print(f"Error: path is not a folder: {folder}", file=sys.stderr)
        return 1

    results, errors = load_results(folder, arguments.recursive)
    best = get_best_results(results)

    print_report(best, results)

    print(
        f"\nProcessed {len(results)} valid result file(s); "
        f"skipped {len(errors)} invalid file(s)."
    )

    if errors:
        print("\nSkipped files:", file=sys.stderr)
        for path, reason in errors:
            print(f"  {path}: {reason}", file=sys.stderr)

    rows = create_output_rows(best)

    if arguments.csv_output:
        csv_path = arguments.csv_output.expanduser().resolve()
        save_csv(rows, csv_path)
        print(f"CSV report saved to: {csv_path}")

    if arguments.json_output:
        json_path = arguments.json_output.expanduser().resolve()
        save_json(best, json_path)
        print(f"JSON report saved to: {json_path}")

    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())


