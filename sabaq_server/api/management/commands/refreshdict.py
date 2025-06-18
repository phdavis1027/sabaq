from django.core.management.base import BaseCommand
from django.conf import settings
from api.lib.free_dictionary_client import FreeDictionaryClient
from api.models import Language

from frenetic import FreNetic

import os


class Command(BaseCommand):
    """
    AIDEV-NOTE: Hello-world Django management command for refresh_dict
    This command serves as a basic template and can be extended to refresh dictionary data
    """
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
        print(fwn.synsets('chien'))
