# stdlib
import random
import warnings
import time
import os

# 3rd party
import pymongo

from transformers import (
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    CamembertTokenizerFast,
    CamembertForTokenClassification,
    TrainingArguments,
    Trainer,
)

import datasets
from datasets import DatasetDict, Dataset, load_metric

import numpy as np

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import torch

from googletrans import Translator

from nltk.translate import meteor_score
import nltk

from sklearn.metrics.pairwise import cosine_similarity

import spacy
import spacy.util as spacy_util

# local

import util


def load_data():
    client = pymongo.MongoClient("localhost", 27017)
    db = client["sabaq"]
    collection = db["training"]

    return collection.find()


# TODO: Next step is to create ONE reference model
# that all different invocations of "model" refer to.
model_name = "camembert/camembert-base-wikipedia-4gb"
bundle = util.BaseModelBundle(model_name, "seqeval")

combined_data = [(it["ex"], it["tag"]) for it in load_data()]

total_samples = len(combined_data)
train_size = int(total_samples * 0.8)
val_size = int(total_samples * 0.1)
test_size = total_samples - train_size - val_size

random.seed(42)

random.shuffle(combined_data)

train_data = combined_data[:train_size]
val_data = combined_data[train_size : train_size + val_size]
test_data = combined_data[train_size + val_size :]

train_tokens, train_tags = zip(*train_data)
val_tokens, val_tags = zip(*val_data)
test_tokens, test_tags = zip(*test_data)

print(f"Train: {len(train_data)} Val: {len(val_data)} Test: {len(test_data)}")

train = Dataset.from_dict(
    {
        "id": list(map(str, range(train_size))),
        "tokens": train_tokens,
        "tags": train_tags,
    }
)
validation = Dataset.from_dict(
    {"id": list(map(str, range(val_size))), "tokens": val_tokens, "tags": val_tags}
)
test = Dataset.from_dict(
    {"id": list(map(str, range(test_size))), "tokens": test_tokens, "tags": test_tags}
)

dataset_dict = DatasetDict({"train": train, "validation": validation, "test": test})


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


torch.manual_seed(42)
# TODO:  What is this pattern called?
tokenize_and_align_inputs_with_bundle = lambda examples: tokenize_and_align_inputs(
    examples, bundle.tokenizer
)
tokenized_datasets = dataset_dict.map(
    tokenize_and_align_inputs_with_bundle, batched=True
)

label_list = ["O", "I"]
nlp = util.load_spacy_model("fr_dep_news_trf")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_bert_embedding(text, model, tokenizer):
    input_ids = tokenizer(text, return_tensors="pt").to(device)

    print(f"[INFO input_ids] {type(input_ids)}")
    print(input_ids)

    with torch.no_grad():
        model_output = model(input_ids)

    embeddings = model_output.last_hidden_state[0]

    return embeddings.mean(dim=0).cpu().numpy()


def compute_semantic_relatedness(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0, 0]


def filter_pos(text, pos_tags_to_keep):
    filtered_words = []

    for word in text:
        for token in nlp(word):
            if token.pos_ in pos_tags_to_keep:
                filtered_words.append(token.text)

    return filtered_words


def calculate_cohesion_score(context_words, idiom_words, model, tokenizer):
    filtered_context_words = filter_pos(context_words, ["NOUN", "VERB"])

    word_embeddings = {}

    for word in filtered_context_words:
        word_embeddings[word] = get_bert_embedding(word, model, tokenizer)

    cohesion_graph = np.zeros(
        (len(filtered_context_words), len(filtered_context_words))
    )

    for i, word1 in enumerate(filtered_context_words):
        for j, word2 in enumerate(filtered_context_words):
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
            word_embeddings[word] = get_bert_embedding(word, model, tokenizer)

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


def meteor(sen, max_retries=3, timeout_seconds=10):
    for retry in range(max_retries):
        try:
            translated_text = translator.translate(
                sen, dest="hi", timeout=timeout_seconds
            ).text
            back_translated_text = translator.translate(
                translated_text, dest="fr", timeout=timeout_seconds
            ).text
            bsen = back_translated_text
            r = [sen.split()]
            c = bsen.split()
            meteor_score_value = meteor_score.meteor_score(r, c)

            return meteor_score_value, sen, bsen
        except Exception as e:
            print(
                f"An error occurred during translation (Retry {retry + 1}/{max_retries}): {e}"
            )
            time.sleep(1)

    print(f"Failed to translate after {max_retries} retries.")
    return 0, None, None


nltk.download("wordnet")
translator = Translator()

backup_bundle = util.BaseModelBundle(
    "camembert/camembert-base-wikipedia-4gb", "seqeval"
)


class IdiomRecognitionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        ip_ids = inputs["input_ids"]
        labels = inputs.pop("labels")

        b = 0
        a = 0
        c = 0

        m, n = idiom_part(ip_ids, labels, self.tokenizer)

        if len(n) != 0:
            c, a, b = calculate_cohesion_score(
                m, n, backup_bundle.model, backup_bundle.tokenizer
            )

        x = " ".join(m)

        y, i, j = meteor(x)

        outputs = model(**inputs)
        logits = outputs.logits

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100
        )

        if y < 0.7:
            loss = loss * 100

        if b - a > 0.02:
            loss = loss * 100

        if return_outputs:
            return loss, outputs
        else:
            return loss


args = TrainingArguments(
    "test-idiom-recognition",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    seed=42,
)


def compute_metrics(preds, metric: datasets.Metric) -> dict:
    pred_logits, labels = preds

    pred_logits = np.argmax(pred_logits, axis=2)

    # Remove ignored index (special tokens)
    predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
        [label_list[l] for (_, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    flat_predictions = [label for sublist in predictions for label in sublist]
    flat_true_labels = [label for sublist in true_labels for label in sublist]

    report = classification_report(flat_true_labels, flat_predictions, digits=4)

    print(report)

    results = metric.compute(predictions=predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


compute_metrics_with_bundle = lambda preds: compute_metrics(preds, bundle.metric)
trainer = IdiomRecognitionTrainer(
    model=bundle.model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=bundle.collator,
    tokenizer=bundle.tokenizer,
    compute_metrics=compute_metrics_with_bundle,
)


trainer.train()
