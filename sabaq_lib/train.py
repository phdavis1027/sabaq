# stdlib
import random

# 3rd party
import pymongo

from datasets import DatasetDict, Dataset
import datasets
import numpy as np
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from sklearn.metrics import classification_report
from datasets import DatasetDict, Dataset
from transformers import TrainingArguments, Trainer
import torch
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import spacy


def load_data():
    client = pymongo.MongoClient("localhost", 27017)
    db = client["sabaq"]
    collection = db["training"]

    return collection.find()


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

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


def tokenize_and_align_inputs(examples, label_all_tokens=True):
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

tokenized_datasets = dataset_dict.map(tokenize_and_align_inputs, batched=True)

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

data_collator = DataCollatorForTokenClassification(tokenizer)

metric = datasets.load_metric("seqeval")

label_list = ["O", "I"]


def compute_metrics(preds):
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


nlp = spacy.load("fr_deps_news_trf")


def get_bert_embedding(text, model, tokenizer):
    input_ids = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        model_output = model(input_ids)

    embeddings = model_output.last_hidden_state[0]

    return embeddings.mean(dim=0).cpu().numpy()


def compute_semantic_relatedness(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0, 0]


def filter_pos(text, pos_tags_to_keep):
    filtered_words = []

    doc = nlp(text)

    for token in doc:
        if token.pos_ in pos_tags_to_keep:
            filtered_words.append(token.text)


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
