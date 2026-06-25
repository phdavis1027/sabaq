from itemadapter import ItemAdapter


class StatisticsPipeline:
    def open_spider(self, spider):
        self.stats = {
            'unique_idioms': 0,
            'total_examples': 0,
            'avg_examples_per_idiom': 0.0,
            'avg_length_of_example': 0.0,
        }

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)

        self.stats['unique_idioms'] += 1
        self.stats['total_examples'] += len(adapter['examples'])
        self.stats['avg_length_of_example'] += sum(len(example) for example in adapter['examples'])
        self.stats['avg_examples_per_idiom'] = self.stats['total_examples'] / self.stats['unique_idioms']

        return item

    def close_spider(self, spider):
        if self.stats['total_examples']:
            self.stats['avg_length_of_example'] /= self.stats['total_examples']

        print(self.stats)
