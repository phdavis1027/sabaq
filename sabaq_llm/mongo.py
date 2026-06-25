from __future__ import annotations

from typing import Any, Iterable


def training_rows_from_record(record: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield training_row_from_tokens_and_labels(
        record,
        record.get("tokens"),
        record.get("labels"),
    )


def training_row_from_tokens_and_labels(
    record: dict[str, Any],
    tokens: list[str] | tuple[str, ...] | None,
    labels: list[int] | tuple[int, ...] | None,
    suffix: str | None = None,
) -> dict[str, Any]:
    if tokens is None or labels is None:
        raise ValueError(f"Training record lacks tokens/labels: {record.get('_id')}")
    if len(tokens) != len(labels):
        raise ValueError(f"Training record has mismatched tokens/labels: {record.get('_id')}")

    row_id = str(record.get("_id"))
    if suffix is not None:
        row_id = f"{row_id}:{suffix}"

    return {
        "id": row_id,
        "tokens": list(tokens),
        "word_labels": [int(label) for label in labels],
        "raw_example": record.get("raw_example"),
        "matched_idioms": record.get("matched_idioms", []),
    }
