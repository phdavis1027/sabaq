from __future__ import annotations

import random
from typing import Any

from datasets import Dataset, DatasetDict


def split_rows(
    rows: list[dict[str, Any]],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("No training rows loaded")

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_size = int(total * train_ratio)
    validation_size = int(total * validation_ratio)

    if total >= 3:
        train_size = max(train_size, 1)
        validation_size = max(validation_size, 1)
        if train_size + validation_size >= total:
            train_size = total - 2
            validation_size = 1

    train = shuffled[:train_size] or shuffled
    validation = shuffled[train_size : train_size + validation_size]
    test = shuffled[train_size + validation_size :]

    return train, validation, test


def build_dataset_dict(
    rows: list[dict[str, Any]],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    train_rows, validation_rows, test_rows = split_rows(
        rows,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    return DatasetDict(
        {
            "train": Dataset.from_list(_dataset_rows(train_rows)),
            "validation": Dataset.from_list(_dataset_rows(validation_rows)),
            "test": Dataset.from_list(_dataset_rows(test_rows)),
        }
    )


def build_tokenized_dataset(
    rows: list[dict[str, Any]],
    tokenizer,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
    label_all_tokens: bool = True,
) -> DatasetDict:
    dataset = build_dataset_dict(
        rows,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    return tokenize_dataset_dict(
        dataset,
        tokenizer,
        label_all_tokens=label_all_tokens,
    )


def tokenize_dataset_dict(
    dataset: DatasetDict,
    tokenizer,
    label_all_tokens: bool = True,
) -> DatasetDict:
    def align(examples):
        return tokenize_and_align_labels(
            examples,
            tokenizer=tokenizer,
            label_all_tokens=label_all_tokens,
        )

    return dataset.map(align, batched=True)


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens: bool = True):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )

    labels = []
    for batch_index, word_labels in enumerate(examples["word_labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
        previous_word_index = None
        label_ids = []

        for word_index in word_ids:
            if word_index is None:
                label_ids.append(-100)
            elif word_index != previous_word_index:
                label_ids.append(word_labels[word_index])
            else:
                label_ids.append(word_labels[word_index] if label_all_tokens else -100)
            previous_word_index = word_index

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def _dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "tokens": row["tokens"],
            "word_labels": row["word_labels"],
            "raw_example": row.get("raw_example"),
            "matched_idioms": row.get("matched_idioms", []),
        }
        for row in rows
    ]