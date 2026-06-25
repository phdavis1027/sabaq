from __future__ import annotations

import os

import pymongo
import spacy
import torch
from transformers import TrainingArguments

from sabaq_llm.data import build_tokenized_dataset
from sabaq_llm.model import load_cohesion_model_bundle, load_token_classification_bundle
from sabaq_llm.mongo import training_rows_from_record
from sabaq_llm.trainer import (
    AsyncRuntime,
    CohesionConfig,
    CohesionScorer,
    IdiomRecognitionTrainer,
    MeteorScorer,
    TranslationMeteorConfig,
    compute_metrics,
)


def main() -> None:
    seed = int(os.getenv("SEED", "42"))
    torch.manual_seed(seed)

    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db = os.getenv("MONGO_DB", "sabaq")
    training_collection = os.getenv("TRAINING_COLLECTION", "training_fr_wiktionary")
    max_training_rows = os.getenv("MAX_TRAINING_ROWS")

    client = pymongo.MongoClient(mongo_uri)
    collection = client[mongo_db][training_collection]
    cursor = collection.find({})
    if max_training_rows:
        cursor = cursor.limit(int(max_training_rows))

    rows = []
    for record in cursor:
        rows.extend(training_rows_from_record(record))

    model_name = os.getenv("MODEL_NAME", "camembert/camembert-base-wikipedia-4gb")
    cohesion_model_name = os.getenv("COHESION_MODEL_NAME", model_name)
    metric_name = os.getenv("METRIC_NAME", "seqeval")
    spacy_model = os.getenv("SPACY_MODEL", "fr_dep_news_trf")
    output_dir = os.getenv("OUTPUT_DIR", "runs/idiom-recognition")

    token_bundle = load_token_classification_bundle(model_name, metric_name=metric_name)
    cohesion_bundle = load_cohesion_model_bundle(cohesion_model_name)
    tokenized_datasets = build_tokenized_dataset(
        rows,
        token_bundle.tokenizer,
        train_ratio=float(os.getenv("TRAIN_RATIO", "0.8")),
        validation_ratio=float(os.getenv("VALIDATION_RATIO", "0.1")),
        seed=seed,
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
        per_device_train_batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "1")),
        per_device_eval_batch_size=int(os.getenv("EVAL_BATCH_SIZE", "1")),
        num_train_epochs=float(os.getenv("NUM_TRAIN_EPOCHS", "3")),
        weight_decay=float(os.getenv("WEIGHT_DECAY", "0.01")),
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
                spacy.load(spacy_model),
                cohesion_bundle.device,
                cohesion_config,
            ),
            translation_config=translation_config,
            cohesion_config=cohesion_config,
            async_runtime=async_runtime,
        )

        trainer.train()
        trainer.eval_dataset = tokenized_datasets["test"]
        trainer.evaluate()
        trainer.save_model(output_dir)
    finally:
        async_runtime.run(meteor_scorer.close())
        async_runtime.close()
        client.close()


if __name__ == "__main__":
    main()
