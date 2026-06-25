from __future__ import annotations

import argparse
import os
from pathlib import Path

from sabaq_llm.data import build_dataset_dict
from sabaq_llm.mongo import load_training_rows_from_mongo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Mongo training records as a Hugging Face DatasetDict."
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("HF_DATASET_OUTPUT_DIR", "runs/hf-training-dataset"),
        help="Directory where the Hugging Face dataset will be saved.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=float(os.getenv("TRAIN_RATIO", "0.8")),
        help="Fraction of rows to use for the train split.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=float(os.getenv("VALIDATION_RATIO", "0.1")),
        help="Fraction of rows to use for the validation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("SEED", "42")),
        help="Random seed for deterministic dataset splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = load_training_rows_from_mongo(
        mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        mongo_db=os.getenv("MONGO_DB", "sabaq"),
        training_collection=os.getenv("TRAINING_COLLECTION", "training_fr_wiktionary"),
        max_training_rows=os.getenv("MAX_TRAINING_ROWS"),
    )
    dataset = build_dataset_dict(
        rows,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    dataset.save_to_disk(str(output_dir))

    split_sizes = ", ".join(
        f"{split}={dataset[split].num_rows}" for split in dataset.keys()
    )
    print(f"Exported Hugging Face dataset: {split_sizes}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
