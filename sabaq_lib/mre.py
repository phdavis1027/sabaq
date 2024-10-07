import torch

import numpy as np

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

import datasets
from datasets import DatasetDict, Dataset, load_metric

import util
from util import (
    BaseModelBundle,
    IdiomRecognitionTrainer,
)

bundle_params = ("camembert/camembert-base-wikipedia-4gb", "seqeval")
bundle = BaseModelBundle(*bundle_params)

tokenized_datasets = util.load_training_data(bundle.tokenizer)


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

label_list = [
    "O",
    "I",
]
nlp = util.load_spacy_model("fr_dep_news_trf")


def compute_metrics(preds, metric: datasets.Metric) -> dict:
    predictions, labels = preds

    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    print(classification_report(true_labels, true_predictions, digits=4))

    results = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


torch.manual_seed(42)

# TODO:  What is this pattern called?
compute_metrics_with_instance = lambda p: compute_metrics(p, bundle.metric)
trainer = IdiomRecognitionTrainer(
    model=bundle.model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=bundle.collator,
    tokenizer=bundle.tokenizer,
    compute_metrics=compute_metrics_with_instance,
)

trainer.train()
