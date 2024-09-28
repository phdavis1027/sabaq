from itemadapter import ItemAdapter

import git

import time

class StatisticsPipeline:
    def open_spider(self, spider):
        repo = git.Repo(search_parent_directories=True) 
        sha = repo.head.object.hexsha

        self.stats = {
            'run_id': f'{spider.name}-{time.time()}-{sha}',
            'unique_idioms': 0,
            'total_examples': 0,
            'avg_examples_per_idiom': 0,
            'avg_length_of_example': 0,
        }

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)

        self.stats['unique_idioms'] += 1
        self.stats['total_examples'] += len(adapter['examples'])
        self.stats['avg_length_of_example'] += sum(len(example) for example in adapter['examples'])
        self.stats['avg_examples_per_idiom'] = self.stats['total_examples'] / self.stats['unique_idioms']

        return item

    def close_spider(self, spider):
        self.stats['avg_length_of_example'] /= self.stats['total_examples']

        print(self.stats)
