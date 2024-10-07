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

from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

from transformers import (
    PreTrainedTokenizerFast,
    PreTrainedModel,
    AutoTokenizer,
    AutoModel,
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


class BackupModelBundle:
    def __init__(self, model_name: str, metric: str) -> None:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        self.model = model
        self.tokenizer = tokenizer


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


def get_bert_embedding(text, model, tokenizer, device):
    print("about to tokenize for bert")
    input_ids = tokenizer(text, return_tensors="pt").to(device)
    print("tokenized for bert")

    with torch.no_grad():
        model_output = model(input_ids)

    embeddings = model_output.last_hidden_state[0]

    return embeddings.mean(dim=0).cpu().numpy()


def compute_semantic_relatedness(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0, 0]


def filter_pos(text, pos_tags_to_keep, nlp):
    filtered_words = []

    for word in text:
        for token in nlp(word):
            if token.pos_ in pos_tags_to_keep:
                filtered_words.append(token.text)

    return filtered_words


def calculate_cohesion_score(context_words, idiom_words, model, tokenizer, nlp, device):
    print("Filtering context words")
    filtered_context_words = filter_pos(context_words, ["NOUN", "VERB"], nlp)

    word_embeddings = {}

    print("Computing embeddings")
    for word in filtered_context_words:
        print(f"Computing embedding for {word}")
        word_embeddings[word] = get_bert_embedding(word, model, tokenizer, device)

    cohesion_graph = np.zeros(
        (len(filtered_context_words), len(filtered_context_words))
    )

    for i, word1 in enumerate(filtered_context_words):
        for j, word2 in enumerate(filtered_context_words):
            print(f"Computing semantic relatedness between {word1} and {word2}")
            cohesion_graph[i, j] = compute_semantic_relatedness(
                word_embeddings[word1], word_embeddings[word2]
            )

    connectivity = cohesion_graph.mean()

    idiom_indices = [filtered_context_words.index(word) for word in idiom_words]

    if idiom_indices:
        cohesion_graph = np.delete(cohesion_graph, idiom_indices, axis=0)
        cohesion_graph = np.delete(cohesion_graph, idiom_indices, axis=1)

        connectivity_without_idiom = cohesion_graph.mean()

        if connectivity_without_idiom < connectivity:
            return "idiom", connectivity, connectivity_without_idiom

        else:
            return "literal", connectivity, connectivity_without_idiom

    else:
        filtered_context_words = context_words

        word_embeddings = {}

        for word in filtered_context_words:
            word_embeddings[word] = get_bert_embedding(word, model, tokenizer, model)

        for word1 in filtered_context_words:
            for word2 in filtered_context_words:
                cohesion_graph[i, j] = compute_semantic_relatedness(
                    word_embeddings[word1], word_embeddings[word2]
                )

        connectivity = cohesion_graph.mean()

        idiom_indices = [
            filtered_context_words.index(word)
            for word in idiom_words
            if word in filtered_context_words
        ]

        if idiom_indices:
            cohesion_graph = np.delete(cohesion_graph, idiom_indices, axis=0)
            cohesion_graph = np.delete(cohesion_graph, idiom_indices, axis=1)

            connectivity_without_idiom = cohesion_graph.mean()

            if connectivity_without_idiom < connectivity:
                return "idiom", connectivity, connectivity_without_idiom

            else:
                return "literal", connectivity, connectivity_without_idiom
        else:
            return "idiom", 0, 0


def idiom_part(ip_ids, labels, tokenizer):
    idiom = []
    for i, l in enumerate(labels.view(-1)):
        if l == 1:
            idiom.append(ip_ids.view(-1)[i])

    return (
        tokenizer.decode(ip_ids.view(-1))
        .replace("[CLS]", "")
        .replace("[SEP]", "")
        .split(),
        tokenizer.decode(idiom).split(),
    )


class IdiomRecognitionTrainer(Trainer):
    def __init__(
        self, cohesion_model, cohesion_tokenizer, nlp, device, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cohesion_model = cohesion_model
        self.cohesion_tokenizer = cohesion_tokenizer
        self.nlp = nlp
        self.device = device

    def compute_loss(self, model, inputs, return_outputs=False):
        ip_ids = inputs["input_ids"]
        labels = inputs.pop("labels")

        a = 0
        b = 0
        c = 0

        m, n = idiom_part(ip_ids, labels, self.tokenizer)

        if len(n) != 0:
            c, a, b = calculate_cohesion_score(
                m,
                n,
                self.cohesion_model,
                self.cohesion_tokenizer,
                self.nlp,
                self.device,
            )

        outputs = model(**inputs)

        logits = outputs.logits

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100
        )

        return (loss, outputs) if return_outputs else loss
