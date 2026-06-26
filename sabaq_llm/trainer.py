from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from googletrans import Translator
import httpx
import nltk
from nltk.translate import meteor_score
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
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
    max_retries: int = 3
    timeout_seconds: int = 10


@dataclass
class CohesionConfig:
    enabled: bool = True
    pos_tags_to_keep: tuple[str, ...] = ("NOUN", "VERB")
    threshold: float = 0.02
    penalty_multiplier: float = 100.0


class AsyncRuntime:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._closed = False

    def run(self, awaitable):
        if self._closed:
            raise RuntimeError("Async runtime is closed")
        return self.loop.run_until_complete(awaitable)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.close()


class MeteorScorer:
    def __init__(self, config: TranslationMeteorConfig) -> None:
        self.config = config
        self.translator = Translator(
            timeout=httpx.Timeout(float(config.timeout_seconds)),
        )
        self.cache: dict[str, tuple[float, str, str]] = {}
        nltk.download("wordnet", quiet=True)

    async def score(self, text: str) -> tuple[float, str, str]:
        if text in self.cache:
            return self.cache[text]

        last_exception: Exception | None = None
        for retry in range(self.config.max_retries):
            try:
                translated = await self._translate(
                    text,
                    src=self.config.source_language,
                    dest=self.config.pivot_language,
                )
                backtranslated = await self._translate(
                    translated,
                    src=self.config.pivot_language,
                    dest=self.config.source_language,
                )
                value = meteor_score.meteor_score([text.split()], backtranslated.split())
                self.cache[text] = (value, text, backtranslated)
                return self.cache[text]
            except Exception as exc:
                last_exception = exc
                print(f"Translation failed on retry {retry + 1}: {exc}")
                await asyncio.sleep(1)

        raise RuntimeError(
            "METEOR backtranslation failed after "
            f"{self.config.max_retries} retries for text: {text!r}"
        ) from last_exception

    async def close(self) -> None:
        await self.translator.client.aclose()

    async def _translate(self, text: str, src: str, dest: str) -> str:
        result = await self.translator.translate(text, src=src, dest=dest)
        return result.text


class CohesionScorer:
    def __init__(self, model, tokenizer, nlp, device, config: CohesionConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.nlp = nlp
        self.device = device
        self.config = config
        self.embedding_cache: dict[str, np.ndarray] = {}

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
        filtered_words = []
        for word in words:
            for token in self.nlp(word):
                if token.pos_ in self.config.pos_tags_to_keep:
                    filtered_words.append(word)
                    break
        return filtered_words

    def _cohesion_graph(self, words: list[str]) -> np.ndarray:
        embeddings = {word: self._embedding(word) for word in words}
        graph = np.zeros((len(words), len(words)))
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                graph[i, j] = cosine_similarity(
                    [embeddings[word1]],
                    [embeddings[word2]],
                )[0, 0]
        return graph

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
        async_runtime: AsyncRuntime,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.idiom_tokenizer = idiom_tokenizer
        self.meteor_scorer = meteor_scorer
        self.cohesion_scorer = cohesion_scorer
        self.async_runtime = async_runtime

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

        meteor_penalty_applies = False
        cohesion_penalty_applies = False
        for sample_input_ids, sample_labels in zip(input_ids, labels):
            context_words, idiom_words = idiom_part(
                sample_input_ids,
                sample_labels,
                self.idiom_tokenizer,
            )
            sentence = " ".join(context_words)

            meteor_config = self.meteor_scorer.config
            if meteor_config.enabled:
                meteor, _, _ = self.async_runtime.run(self.meteor_scorer.score(sentence))
                if meteor < meteor_config.meteor_threshold:
                    meteor_penalty_applies = True

            cohesion_config = self.cohesion_scorer.config
            if cohesion_config.enabled and idiom_words:
                _, connectivity, connectivity_without_idiom = self.cohesion_scorer.calculate(
                    context_words,
                    idiom_words,
                )
                if connectivity_without_idiom - connectivity > cohesion_config.threshold:
                    cohesion_penalty_applies = True

        meteor_config = self.meteor_scorer.config
        if meteor_penalty_applies:
            loss = loss * meteor_config.penalty_multiplier

        cohesion_config = self.cohesion_scorer.config
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

    results = metric.compute(predictions=predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
