from __future__ import annotations

import math
import os

from datasets import DatasetDict, load_dataset, load_from_disk
import spacy
import torch
from transformers import TrainingArguments

from sabaq_llm.data import build_dataset_dict, tokenize_dataset_dict
from sabaq_llm.env import env_bool
from sabaq_llm.model import load_cohesion_model_bundle, load_token_classification_bundle
from sabaq_llm.mongo import load_training_rows_from_mongo
from sabaq_llm.trainer import (
    AsyncRuntime,
    CohesionConfig,
    CohesionScorer,
    IdiomRecognitionTrainer,
    MeteorScorer,
    TranslationMeteorConfig,
    compute_metrics,
)


def load_dataset_dict_from_env(source: str, seed: int) -> DatasetDict:
    match source:
        case "mongo":
            rows = load_training_rows_from_mongo(
                mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
                mongo_db=os.getenv("MONGO_DB", "sabaq"),
                training_collection=os.getenv(
                    "TRAINING_COLLECTION",
                    "training_fr_wiktionary",
                ),
                max_training_rows=os.getenv("MAX_TRAINING_ROWS"),
            )
            dataset = build_dataset_dict(
                rows,
                train_ratio=float(os.getenv("TRAIN_RATIO", "0.8")),
                validation_ratio=float(os.getenv("VALIDATION_RATIO", "0.1")),
                seed=seed,
            )
        case "hf_disk":
            hf_dataset_path = os.getenv("HF_DATASET_PATH")
            if not hf_dataset_path:
                raise ValueError("HF_DATASET_PATH is required when TRAINING_DATA_SOURCE=hf_disk")
            dataset = load_from_disk(hf_dataset_path)
        case "hf_hub":
            hf_dataset_repo = os.getenv("HF_DATASET_REPO")
            if not hf_dataset_repo:
                raise ValueError("HF_DATASET_REPO is required when TRAINING_DATA_SOURCE=hf_hub")
            dataset = load_dataset(hf_dataset_repo)
        case _:
            raise ValueError(
                "TRAINING_DATA_SOURCE must be one of: mongo, hf_disk, hf_hub"
            )

    if not isinstance(dataset, DatasetDict):
        raise ValueError("Expected a Hugging Face DatasetDict with train/validation/test splits")

    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits.difference(dataset.keys())
    if missing_splits:
        raise ValueError(f"Dataset is missing required splits: {sorted(missing_splits)}")

    return dataset


def print_training_summary(
    dataset: DatasetDict,
    training_data_source: str,
    train_batch_size: int,
    eval_batch_size: int,
    num_train_epochs: float,
    gradient_accumulation_steps: int,
) -> None:
    train_examples = dataset["train"].num_rows
    validation_examples = dataset["validation"].num_rows
    test_examples = dataset["test"].num_rows
    train_batches_per_epoch = math.ceil(train_examples / train_batch_size)
    optimizer_steps_per_epoch = math.ceil(
        train_batches_per_epoch / gradient_accumulation_steps
    )
    total_optimizer_steps = math.ceil(optimizer_steps_per_epoch * num_train_epochs)

    print("Training setup:", flush=True)
    print(f"  data source: {training_data_source}", flush=True)
    print(
        "  split sizes: "
        f"train={train_examples}, validation={validation_examples}, test={test_examples}",
        flush=True,
    )
    print(
        "  batches/steps: "
        f"train_batch_size={train_batch_size}, "
        f"eval_batch_size={eval_batch_size}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}",
        flush=True,
    )
    print(
        "  estimated training progress: "
        f"{train_batches_per_epoch} batches/epoch, "
        f"{optimizer_steps_per_epoch} optimizer steps/epoch, "
        f"{total_optimizer_steps} total optimizer steps over {num_train_epochs:g} epochs",
        flush=True,
    )


def main() -> None:
    seed = int(os.getenv("SEED", "42"))
    torch.manual_seed(seed)

    training_data_source = os.getenv("TRAINING_DATA_SOURCE", "mongo")
    dataset = load_dataset_dict_from_env(training_data_source, seed)

    model_name = os.getenv("MODEL_NAME", "camembert/camembert-base-wikipedia-4gb")
    cohesion_model_name = os.getenv("COHESION_MODEL_NAME", model_name)
    metric_name = os.getenv("METRIC_NAME", "seqeval")
    spacy_model = os.getenv("SPACY_MODEL", "fr_dep_news_trf")
    output_dir = os.getenv("OUTPUT_DIR", "runs/idiom-recognition")

    token_bundle = load_token_classification_bundle(model_name, metric_name=metric_name)
    cohesion_bundle = load_cohesion_model_bundle(cohesion_model_name)
    tokenized_datasets = tokenize_dataset_dict(dataset, token_bundle.tokenizer)

    train_batch_size = int(os.getenv("TRAIN_BATCH_SIZE", "1"))
    eval_batch_size = int(os.getenv("EVAL_BATCH_SIZE", "1"))
    num_train_epochs = float(os.getenv("NUM_TRAIN_EPOCHS", "3"))
    gradient_accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1"))
    print_training_summary(
        tokenized_datasets,
        training_data_source=training_data_source,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    translation_config = TranslationMeteorConfig(
        enabled=os.getenv("TRANSLATION_METEOR_ENABLED", "1") == "1",
        source_language=os.getenv("SOURCE_LANGUAGE", "fr"),
        pivot_language=os.getenv("PIVOT_LANGUAGE", "hi"),
        meteor_threshold=float(os.getenv("METEOR_THRESHOLD", "0.7")),
        penalty_multiplier=float(os.getenv("OBJECTIVE_PENALTY_MULTIPLIER", "100")),
        max_retries=int(os.getenv("TRANSLATION_MAX_RETRIES", "3")),
        timeout_seconds=int(os.getenv("TRANSLATION_TIMEOUT_SECONDS", "10")),
    )
    cohesion_config = CohesionConfig(
        enabled=os.getenv("COHESION_ENABLED", "1") == "1",
        threshold=float(os.getenv("COHESION_THRESHOLD", "0.02")),
        penalty_multiplier=float(os.getenv("OBJECTIVE_PENALTY_MULTIPLIER", "100")),
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=float(os.getenv("LEARNING_RATE", "2e-5")),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=float(os.getenv("WEIGHT_DECAY", "0.01")),
        logging_strategy="steps",
        logging_steps=int(os.getenv("LOGGING_STEPS", "10")),
        logging_first_step=True,
        disable_tqdm=env_bool("DISABLE_TQDM", False),
        seed=seed,
    )

    async_runtime = AsyncRuntime()
    meteor_scorer = MeteorScorer(translation_config)
    try:
        trainer = IdiomRecognitionTrainer(
            model=token_bundle.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=token_bundle.data_collator,
            processing_class=token_bundle.tokenizer,
            compute_metrics=lambda preds: compute_metrics(preds, token_bundle.metric),
            idiom_tokenizer=token_bundle.tokenizer,
            meteor_scorer=meteor_scorer,
            cohesion_scorer=CohesionScorer(
                cohesion_bundle.model,
                cohesion_bundle.tokenizer,
                spacy.load(spacy_model, disable=["parser", "ner"]),
                cohesion_bundle.device,
                cohesion_config,
            ),
            async_runtime=async_runtime,
        )

        trainer.train()
        trainer.eval_dataset = tokenized_datasets["test"]
        trainer.evaluate()
        trainer.save_model(output_dir)
    finally:
        async_runtime.run(meteor_scorer.close())
        async_runtime.close()



if __name__ == "__main__":
    main()
