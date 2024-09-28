import unicodedata

import scrapy

import git

from bs4 import BeautifulSoup as bs

# TODO: Add a pipeline step that applies unicode normalization

IDIOTISMES_ANIMALIERS = "https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Idiotismes_animaliers_en_fran%C3%A7ais"
IDIOTISMES_DEMONYMES = "https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Idiotismes_avec_d%C3%A9monymes_en_fran%C3%A7ais"
FAUX_PROVERBES = (
    "https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Faux_proverbes_en_fran%C3%A7ais"
)
IDIOTISMES_AVEC_PRENOMS = "https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Idiotismes_avec_pr%C3%A9noms_en_fran%C3%A7ais"
IDIOTISMES_AVEC_TOPONYMES = "https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Idiotismes_avec_toponymes_en_fran%C3%A7ais"
IDIOTISMES_BOTANIQUES = "https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Idiotismes_botaniques_en_fran%C3%A7ais"
IDIOTISMES_CHROMATIQUE = "https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Idiotismes_chromatiques_en_fran%C3%A7ais"
IDIOTISMES_CORPORELS = "https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Idiotismes_corporels_en_fran%C3%A7ais"
IDIOTISMES_GASTRONOMIQUES = "https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Idiotismes_gastronomiques_en_fran%C3%A7ais"
IDIOTISMES_MINERAUX = "https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Idiotismes_min%C3%A9raux_en_fran%C3%A7ais"
IDIOTISMES_NUMERIQUES = "https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Idiotismes_num%C3%A9riques_en_fran%C3%A7ais"
IDIOTISMES_VESTIMENTAIRES = "https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Idiotismes_vestimentaires_en_fran%C3%A7ais"
PROVERBES = "https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Proverbes_en_fran%C3%A7ais"
COMAPRAISONS = (
    "https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Comparaisons_en_fran%C3%A7ais"
)
IDIOTISMES_REDONDANTES = "https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Idiotismes_redondantes_en_fran%C3%A7ais"
IDEMS_DE_POLARITE_NEGATIVE = "https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Id%C3%A9es_de_polarit%C3%A9_n%C3%A9gative_en_fran%C3%A7ais"
LOCUTIONS_VERBALES = (
    "https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Locutions_verbales_en_fran%C3%A7ais"
)

urls = [
    IDIOTISMES_ANIMALIERS,
    IDIOTISMES_DEMONYMES,
    FAUX_PROVERBES,
    IDIOTISMES_AVEC_PRENOMS,
    IDIOTISMES_AVEC_TOPONYMES,
    IDIOTISMES_BOTANIQUES,
    IDIOTISMES_CHROMATIQUE,
    IDIOTISMES_CORPORELS,
    IDIOTISMES_GASTRONOMIQUES,
    IDIOTISMES_MINERAUX,
    IDIOTISMES_NUMERIQUES,
    IDIOTISMES_VESTIMENTAIRES,
    PROVERBES,
    COMAPRAISONS,
    IDIOTISMES_REDONDANTES,
    IDEMS_DE_POLARITE_NEGATIVE,
]


class FrWiktionarySpider(scrapy.Spider):
    name = "fr_wiktionary"

    def __init__(self, category=None, *args, **kwargs):
        super(FrWiktionarySpider, self).__init__(*args, **kwargs)
        self.run_id = kwargs["run_id"]

    def start_requests(self):
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for entry in response.css(".mw-category-group li"):
            yield response.follow(
                entry.css("a::attr(href)").extract_first(), callback=self.parse_entry
            )

        next_page = response.css(".mw-category-group a::attr(href)").extract_first()

        if next_page is not None:
            yield response.follow(next_page, self.parse)

    def parse_entry(self, response):
        # Iterate over the definitions
        idiom = response.css(".mw-page-title-main::text").extract_first()

        soup = bs(response.body, "html.parser")

        examples = []

        EXAMPLE_SELECTOR = "span.example > q > bdi.lang-fr"

        for example in soup.select(EXAMPLE_SELECTOR):
            example = example.get_text(separator=" ").strip()
            example = unicodedata.normalize("NFKC", example)

            examples.append(example)

        yield {"run": self.run_id, "idiom": idiom, "examples": examples}
