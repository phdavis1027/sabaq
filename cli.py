# stdlib
import sys
import os


# 3rd party
from scrapy.crawler import CrawlerProcess
from scrapy.settings import Settings

# Local
from scrape_tools.spiders.fr_wiktionary_spider import FrWiktionarySpider

from sabaq_lib.util import gen_run_id

if sys.argv[1] == 'run-spider':
    run_id = gen_run_id('fr_wiktionary', 'wiktionary', 'fr')

    settings = Settings()
    os.environ['SCRAPY_SETTINGS_MODULE'] = 'scrape_tools.settings'
    settings_module_path = os.environ['SCRAPY_SETTINGS_MODULE']
    settings.setmodule(settings_module_path, priority='project')

    process = CrawlerProcess(settings=settings)
    process.crawl(FrWiktionarySpider, run_id=run_id)

    process.start()


