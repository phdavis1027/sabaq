from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse
import json

User = get_user_model()


class Command(BaseCommand):
    help = '''Query user dictionary entries by hitting the actual API endpoint

    This command provides a transparent CLI interface to test the user_dictionary_entries API endpoint
    for desk checking purposes by making actual HTTP requests.

    Examples:
      python manage.py dict_entry_query --user 1
      python manage.py dict_entry_query --user 2 --languages fr,en
      python manage.py dict_entry_query --user 3 --order_bys word,-language
      python manage.py dict_entry_query --user 4 --languages fr --order_bys -word,language
    '''

    def add_arguments(self, parser):
        """Add command line arguments"""
        parser.add_argument(
            '--user',
            type=int,
            required=True,
            help='User ID to query dictionary entries for'
        )
        parser.add_argument(
            '--languages',
            type=str,
            default=None,
            help='Comma-separated list of language codes to filter by (e.g., "fr,en")'
        )
        parser.add_argument(
            '--order_bys',
            type=str,
            default=None,
            help='Comma-separated list of fields to order by (e.g., "word,-language")'
        )

    def handle(self, *args, **options):
        try:
            user = User.objects.get(id=options['user'])
        except User.DoesNotExist:
            self.stdout.write(
                self.style.ERROR(f'User with ID {options["user"]} does not exist')
            )
            return

        # Create test client and authenticate user
        client = Client()
        client.force_login(user)

        # Build query parameters
        params = {}
        if options['languages']:
            params['languages'] = options['languages']
            self.stdout.write(f'Filtering by languages: {options["languages"]}')

        if options['order_bys']:
            params['order_bys'] = options['order_bys']
            self.stdout.write(f'Ordering by: {options["order_bys"]}')

        # Make request to the actual endpoint
        try:
            response = client.get('/api/dictionary_entries', params)

            self.stdout.write(f'Request made for user ID {options["user"]} ({user.username})')
            self.stdout.write(f'Response status: {response.status_code}')

            if response.status_code == 200:
                # Parse and display the response
                response_data = response.json()
                self.stdout.write(json.dumps(response_data, indent=2))
            else:
                self.stdout.write(
                    self.style.ERROR(f'Request failed with status {response.status_code}')
                )
                self.stdout.write(response.content.decode())

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error making request: {str(e)}')
            )
