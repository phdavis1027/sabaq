# stdlib

import time
import os
import sys

# 3rd party

import git

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
)

import datasets

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
        print("loading model")
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=2
        )

        print("loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("checking and setting device")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print("loading data collator")
        collator = DataCollatorForTokenClassification(tokenizer)

        print("loading metric")
        metric = datasets.load_metric(metric)

        self.model = model
        self.tokenizer = tokenizer
        self.collator = collator
        self.metric = metric
