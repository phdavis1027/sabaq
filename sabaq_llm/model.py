from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import datasets
import torch
from transformers import (
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)




@dataclass
class TokenClassificationBundle:
    model: Any
    tokenizer: Any
    data_collator: Any
    metric: Any
    device: torch.device


@dataclass
class CohesionModelBundle:
    model: Any
    tokenizer: Any
    device: torch.device


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_token_classification_bundle(
    model_name: str,
    metric_name: str = "seqeval",
    num_labels: int = 2,
) -> TokenClassificationBundle:
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    model.to(device)
    return TokenClassificationBundle(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        metric=datasets.load_metric(metric_name),
        device=device,
    )


def load_cohesion_model_bundle(model_name: str) -> CohesionModelBundle:
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    return CohesionModelBundle(model=model, tokenizer=tokenizer, device=device)
