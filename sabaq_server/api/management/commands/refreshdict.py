from django.core.management.base import BaseCommand
from django.conf import settings
from api.lib.free_dictionary_client import FreeDictionaryClient
from api.models import Language, DictionaryEntry, Definition, FrenchDefinitionSource

from frenetic import FreNetic

import os


class Command(BaseCommand):
    help = 'Hello world command for refreshing dictionary data'

    def add_arguments(self, parser):
        """Add command line arguments"""
        parser.add_argument(
            '--word',
            type=str,
            default=None,
            help='Refresh all definitions in the dictionary, or just `--word` if given'
        )

    def handle(self, *args, **options):
        fwn = FreNetic(os.path.join(settings.BASE_DIR,
                       'api/static/wolf-1.0b4.xml'))
        # Create Definitions for each DictionaryEntry
        entries = DictionaryEntry.objects.all()
        for entry in entries:
            Definition.objects.create(
                dictionary_entry=entry,
                source=FrenchDefinitionSource.WOLF_WORDNET,
                confidence=1.0,
                usage_count=0
            )
