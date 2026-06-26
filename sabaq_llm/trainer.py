from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nltk
from nltk.translate import meteor_score
import numpy as np
from sklearn.metrics import classification_report
import torch
from transformers import Trainer


LABEL_LIST = ["O", "IDIOM"]


@dataclass
class TranslationMeteorConfig:
    enabled: bool = True
    source_language: str = "fr"
    pivot_language: str = "hi"
    meteor_threshold: float = 0.7
    penalty_multiplier: float = 100.0
    max_input_tokens: int = 256


@dataclass
class CohesionConfig:
    enabled: bool = True
    pos_tags_to_keep: tuple[str, ...] = ("NOUN", "VERB")
    threshold: float = 0.02
    penalty_multiplier: float = 100.0


NLLB_LANG_CODES = {"fr": "fra_Latn", "hi": "hin_Deva"}


class MeteorScorer:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        config: TranslationMeteorConfig,
        cache_path: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.cache: dict[str, float] = {}
        self.cache_path = Path(cache_path) if cache_path else None
        self._load_cache()
        nltk.download("wordnet", quiet=True)

    def _load_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        with self.cache_path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self.cache[entry["text"]] = entry["score"]

    def _persist(self, text: str, score: float) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a") as handle:
            handle.write(json.dumps({"text": text, "score": score}) + "\n")

    def score_batch(self, texts: list[str]) -> list[float]:
        missing = [text for text in dict.fromkeys(texts) if text not in self.cache]
        if missing:
            pivots = self._translate(
                missing, self.config.source_language, self.config.pivot_language
            )
            backtranslated = self._translate(
                pivots, self.config.pivot_language, self.config.source_language
            )
            for text, back in zip(missing, backtranslated):
                score = meteor_score.meteor_score([text.split()], back.split())
                self.cache[text] = score
                self._persist(text, score)
        return [self.cache[text] for text in texts]

    def _translate(self, texts: list[str], src: str, dest: str) -> list[str]:
        self.tokenizer.src_lang = NLLB_LANG_CODES[src]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_input_tokens,
        ).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(NLLB_LANG_CODES[dest]),
                max_length=self.config.max_input_tokens,
            )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


class CohesionScorer:
    def __init__(self, model, tokenizer, nlp, device, config: CohesionConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.nlp = nlp
        self.device = device
        self.config = config
        self.embedding_cache: dict[str, np.ndarray] = {}
        self.nlp_cache: dict[str, bool] = {}

    def calculate(self, context_words: list[str], idiom_words: list[str]):
        filtered_context_words = self._filter_pos(context_words)
        result = self._calculate_for_words(filtered_context_words, idiom_words)
        if result is not None:
            return result

        result = self._calculate_for_words(context_words, idiom_words)
        if result is not None:
            return result

        return "idiom", 0.0, 0.0

    def _calculate_for_words(
        self,
        context_words: list[str],
        idiom_words: list[str],
    ) -> tuple[str, float, float] | None:
        if not context_words:
            return None

        cohesion_graph = self._cohesion_graph(context_words)
        connectivity = float(np.mean(cohesion_graph))
        idiom_indices = [
            context_words.index(word)
            for word in idiom_words
            if word in context_words
        ]

        if not idiom_indices:
            return None

        cohesion_graph = np.delete(cohesion_graph, idiom_indices, axis=0)
        cohesion_graph = np.delete(cohesion_graph, idiom_indices, axis=1)
        connectivity_without_idiom = float(np.mean(cohesion_graph))

        if connectivity_without_idiom > connectivity:
            return "idiom", connectivity, connectivity_without_idiom
        return "literal", connectivity, connectivity_without_idiom

    def _filter_pos(self, words: list[str]) -> list[str]:
        missing = [word for word in dict.fromkeys(words) if word not in self.nlp_cache]
        for word, doc in zip(missing, self.nlp.pipe(missing)):
            self.nlp_cache[word] = any(
                token.pos_ in self.config.pos_tags_to_keep for token in doc
            )
        return [word for word in words if self.nlp_cache[word]]

    def _cohesion_graph(self, words: list[str]) -> np.ndarray:
        embeddings = np.stack([self._embedding(word) for word in words])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = embeddings / norms
        return normalized @ normalized.T

    def _embedding(self, text: str) -> np.ndarray:
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
        embedding = output.last_hidden_state[0].mean(dim=0).cpu().numpy()
        self.embedding_cache[text] = embedding
        return embedding


class IdiomRecognitionTrainer(Trainer):
    def __init__(
        self,
        *args,
        idiom_tokenizer,
        meteor_scorer: MeteorScorer,
        cohesion_scorer: CohesionScorer,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.idiom_tokenizer = idiom_tokenizer
        self.meteor_scorer = meteor_scorer
        self.cohesion_scorer = cohesion_scorer

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        input_ids = inputs["input_ids"]
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=-100,
        )

        if model.training:
            meteor_penalty_applies = False
            cohesion_penalty_applies = False
            sentences: list[str] = []
            for sample_input_ids, sample_labels in zip(input_ids, labels):
                context_words, idiom_words = idiom_part(
                    sample_input_ids,
                    sample_labels,
                    self.idiom_tokenizer,
                )
                sentences.append(" ".join(context_words))

                cohesion_config = self.cohesion_scorer.config
                if cohesion_config.enabled and idiom_words:
                    _, connectivity, connectivity_without_idiom = self.cohesion_scorer.calculate(
                        context_words,
                        idiom_words,
                    )
                    if connectivity_without_idiom - connectivity > cohesion_config.threshold:
                        cohesion_penalty_applies = True

            meteor_config = self.meteor_scorer.config
            if meteor_config.enabled:
                scores = self.meteor_scorer.score_batch(sentences)
                if any(value < meteor_config.meteor_threshold for value in scores):
                    meteor_penalty_applies = True

            if meteor_penalty_applies:
                loss = loss * meteor_config.penalty_multiplier

            if cohesion_penalty_applies:
                loss = loss * cohesion_config.penalty_multiplier
        return (loss, outputs) if return_outputs else loss


def idiom_part(input_ids, labels, tokenizer) -> tuple[list[str], list[str]]:
    flat_input_ids = input_ids.view(-1)
    flat_labels = labels.view(-1)
    idiom_ids = [
        int(flat_input_ids[index])
        for index, label in enumerate(flat_labels)
        if int(label) == 1
    ]
    context = tokenizer.decode(flat_input_ids, skip_special_tokens=True).split()
    idiom = tokenizer.decode(idiom_ids, skip_special_tokens=True).split()
    return context, idiom


def _to_bio(tags: list[str]) -> list[str]:
    bio = []
    previous = "O"
    for tag in tags:
        if tag == "O":
            bio.append("O")
        elif previous == "IDIOM":
            bio.append("I-IDIOM")
        else:
            bio.append("B-IDIOM")
        previous = tag
    return bio


def compute_metrics(eval_preds, metric: Any) -> dict[str, float]:
    pred_logits, labels = eval_preds
    pred_logits = np.argmax(pred_logits, axis=2)

    predictions = [
        [LABEL_LIST[prediction] for prediction, label in zip(prediction_row, label_row) if label != -100]
        for prediction_row, label_row in zip(pred_logits, labels)
    ]
    true_labels = [
        [LABEL_LIST[label] for prediction, label in zip(prediction_row, label_row) if label != -100]
        for prediction_row, label_row in zip(pred_logits, labels)
    ]

    flat_predictions = [label for row in predictions for label in row]
    flat_true_labels = [label for row in true_labels for label in row]
    print(classification_report(flat_true_labels, flat_predictions, digits=4))

    results = metric.compute(
        predictions=[_to_bio(row) for row in predictions],
        references=[_to_bio(row) for row in true_labels],
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
