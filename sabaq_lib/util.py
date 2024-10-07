# stdlib

import time
import os
import sys

# 3rd party

import git

import pymongo

import torch

import spacy
from spacy.language import Language as SpacyLanguage
import spacy.util as spacy_util

from transformers import (
    PreTrainedTokenizerFast,
    PreTrainedModel,
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
)

import datasets
from datasets import Dataset, DatasetDict

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha


def gen_run_id(module: str, source: str, language: str) -> str:
    return f"{module}-{source}-{language}-{sha}-{int(time.time())}"


def find_path_to_site_packages() -> str | None:
    for p in sys.path:
        if "site-packages" in p:
            return p
    return None


def load_spacy_model(model: str) -> SpacyLanguage:
    version = spacy_util.get_package_version(model)
    cfg_dir = f"{model}-{version}"

    spacy_model_path = os.path.join(find_path_to_site_packages(), model, cfg_dir)

    return spacy.load(spacy_model_path)


class BaseModelBundle:
    def __init__(self, model_name: str, metric: str) -> None:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=2
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        collator = DataCollatorForTokenClassification(tokenizer)
        metric = datasets.load_metric(metric)

        self.model = model
        self.tokenizer = tokenizer
        self.collator = collator
        self.metric = metric


def _fetch_from_mongo():
    client = pymongo.MongoClient("localhost", 27017)
    db = client["sabaq"]
    collection = db["training"]

    return collection.find()


def tokenize_and_align_inputs(examples, tokenizer, label_all_tokens=True):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def load_training_data(
    tokenizer, portion_train=0.8, portion_val=0.1
) -> datasets.DatasetDict:
    data = [(it["ex"], it["tag"]) for it in _fetch_from_mongo()]

    n = len(data)
    n_train = int(n * portion_train)
    n_val = int(n * portion_val)
    n_test = n - n_train - n_val

    train_data = data[:n_train]
    val_data = data[n_train : n_train + n_val]
    test_data = data[n_train + n_val :]

    train_tokens, train_tags = zip(*train_data)
    val_tokens, val_tags = zip(*val_data)
    test_tokens, test_tags = zip(*test_data)

    train = Dataset.from_dict(
        {
            "id": list(map(str, range(n_train))),
            "tokens": list(train_tokens),
            "tags": list(train_tags),
        }
    )

    val = Dataset.from_dict(
        {
            "id": list(map(str, range(n_val))),
            "tokens": list(val_tokens),
            "tags": list(val_tags),
        }
    )

    test = Dataset.from_dict(
        {
            "id": list(map(str, range(n_test))),
            "tokens": list(test_tokens),
            "tags": list(test_tags),
        }
    )

    ddict = DatasetDict({"train": train, "validation": val, "test": test})

    tokenize_and_align_inputs_with_bundle = lambda examples: tokenize_and_align_inputs(
        examples, tokenizer
    )
    tokenized_datasets = ddict.map(tokenize_and_align_inputs_with_bundle, batched=True)

    return tokenized_datasets


class IdiomRecognitionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100
        )

        return (loss, outputs) if return_outputs else loss
