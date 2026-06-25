from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from math import ceil
import unicodedata

from itemadapter import ItemAdapter
import pymongo
from rapidfuzz.distance import LCSseq
import spacy


@dataclass
class TrainingExample:
    tokens: list[str]
    labels: list[int]
    raw_example: str
    matched_idioms: set[str] = field(default_factory=set)


class FrLabelIdiomsPipeline:
    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            similarity_threshold=crawler.settings.getfloat("SIMILARITY_THRESHOLD"),
            throw_out_untagged=crawler.settings.getbool("THROW_OUT_UNTAGGED"),
            spacy_model=crawler.settings.get("SPACY_MODEL"),
            mongo_uri=crawler.settings.get("MONGO_URI"),
            mongo_db=crawler.settings.get("MONGO_DB"),
            raw_collection=crawler.settings.get("MONGO_RAW_COLLECTION"),
            training_collection=crawler.settings.get("MONGO_TRAINING_COLLECTION"),
        )

    def __init__(
        self,
        similarity_threshold: float,
        throw_out_untagged: bool,
        spacy_model: str,
        mongo_uri: str,
        mongo_db: str,
        raw_collection: str,
        training_collection: str,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.throw_out_untagged = throw_out_untagged
        self.spacy_model = spacy_model
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self.raw_collection_name = raw_collection
        self.training_collection_name = training_collection
        self.nlp = spacy.load(self.spacy_model)
        self.examples: OrderedDict[tuple[str, ...], TrainingExample] = OrderedDict()

    def open_spider(self, spider):
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.mongo_db]
        self.raw_collection = self.db[self.raw_collection_name]
        self.training_collection = self.db[self.training_collection_name]
        self.raw_collection.delete_many({"run": spider.run_id})
        self.training_collection.delete_many({})

    def close_spider(self, spider):
        try:
            docs = []
            for example in self.examples.values():
                if self.throw_out_untagged and not any(example.labels):
                    continue
                if len(example.tokens) != len(example.labels):
                    raise ValueError("Token and label lengths do not match")
                docs.append(
                    {
                        "run": spider.run_id,
                        "source": spider.name,
                        "language": "fr",
                        "tokens": example.tokens,
                        "labels": example.labels,
                        "raw_example": example.raw_example,
                        "matched_idioms": sorted(example.matched_idioms),
                    }
                )

            if docs:
                self.training_collection.insert_many(docs)
        finally:
            self.client.close()

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        raw = {
            "run": adapter.get("run", spider.run_id),
            "source": spider.name,
            "language": "fr",
            "idiom": unicodedata.normalize("NFKC", adapter.get("idiom", "") or ""),
            "examples": [
                unicodedata.normalize("NFKC", example or "")
                for example in adapter.get("examples", [])
            ],
        }
        self.raw_collection.insert_one(raw)
        self._add_training_examples(raw)
        return item

    def _add_training_examples(self, entry: dict) -> None:
        idiom_tokens = self._lemmatize(entry["idiom"])
        if not idiom_tokens:
            return

        for raw_example in entry["examples"]:
            tokens = self._lemmatize(raw_example)
            if not tokens:
                continue

            key = tuple(tokens)
            if key not in self.examples:
                self.examples[key] = TrainingExample(
                    tokens=tokens,
                    labels=[0] * len(tokens),
                    raw_example=raw_example,
                )

            example = self.examples[key]
            if self._label_idiom(example, entry["idiom"], idiom_tokens):
                example.matched_idioms.add(entry["idiom"])

    def _label_idiom(
        self,
        example: TrainingExample,
        raw_idiom: str,
        idiom_tokens: list[str],
    ) -> bool:
        minimum_matches = ceil(self.similarity_threshold * len(idiom_tokens))
        matched_token_count = LCSseq.similarity(
            example.tokens,
            idiom_tokens,
            score_cutoff=minimum_matches,
        )
        if matched_token_count < minimum_matches:
            return False

        for opcode in LCSseq.opcodes(example.tokens, idiom_tokens):
            if opcode.tag != "equal":
                continue
            example.labels[opcode.src_start : opcode.src_end] = [1] * (
                opcode.src_end - opcode.src_start
            )
        return True

    def _lemmatize(self, text: str) -> list[str]:
        doc = self.nlp(unicodedata.normalize("NFKC", text or ""))
        return [
            token.lemma_
            for token in doc
            if not token.is_punct and not token.is_space
        ]