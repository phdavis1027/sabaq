# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter

from difflib import SequenceMatcher

import spacy

from scrapy.item import Item, Field

class FrLabelIdiomsItem(Item):
    i_training_format = Field()

# TODO: Make this thing dispatch to the correct data source
# IE, we should probably pass some metadata about the source
class FrLabelIdiomsPipeline:
    def __init__(self):
        self.nlp = spacy.load("fr_dep_news_trf")

    # TODO: Implement other similarity metrics
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)

        _idiom = self.nlp(adapter['idiom'])
        idiom  = [token.lemma_ for token in _idiom if not token.is_stop]
        idiom_len = len(idiom)

        i_training_format = []

        for _example in adapter['examples']:
            _example = self.nlp(_example)
            example = [token.lemma_ for token in _example if not token.is_stop]

            tags = [0] * len(example)

            untouched = True 

            for i in range(len(example) - idiom_len + 1):
                current_window = example[i:i + idiom_len]

                similarity = SequenceMatcher(None, idiom, current_window).ratio()

                if similarity > spider.settings.getfloat('SIMILARITY_THRESHOLD'):
                    tags[i:i + idiom_len] = [1] * idiom_len

                    untouched = False

            if untouched and spider.settings.getboolean('THROW_OUT_UNTAGGED'):
                continue

            i_training_format.append({
                'example': example,
                'tags': tags
            })

        return FrLabelIdiomsItem(
            i_training_format=i_training_format,
        )
