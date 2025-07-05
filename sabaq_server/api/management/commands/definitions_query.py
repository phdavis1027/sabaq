from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse
import json

User = get_user_model()


class Command(BaseCommand):
    help = '''Query definitions by hitting the actual API endpoint

    This command provides a transparent CLI interface to test the definitions API endpoint
    for desk checking purposes by making actual HTTP requests.

    Examples:
      python manage.py definitions_query --user 1 --dictionary_entries "word1,word2"
      python manage.py definitions_query --user 2 --dictionary_entries "bonjour,merci" --source wordnet
      python manage.py definitions_query --user 3 --dictionary_entries "chat" --confidence_gt 0.5
      python manage.py definitions_query --user 4 --dictionary_entries "eau,feu" --confidence_gt 0.3 --confidence_lt 0.8
    '''

    def add_arguments(self, parser):
        """Add command line arguments"""
        parser.add_argument(
            '--user',
            type=int,
            required=True,
            help='User ID to query definitions for'
        )
        parser.add_argument(
            '--dictionary_entries',
            type=str,
            required=True,
            help='Comma-separated list of dictionary entry words to get definitions for'
        )
        parser.add_argument(
            '--source',
            type=str,
            default=None,
            help='Filter definitions by source (e.g., "wordnet")'
        )
        parser.add_argument(
            '--confidence_gt',
            type=float,
            default=None,
            help='Filter definitions with confidence greater than this value (0-1)'
        )
        parser.add_argument(
            '--confidence_lt',
            type=float,
            default=None,
            help='Filter definitions with confidence less than this value (0-1)'
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

        # Parse dictionary entries
        dictionary_entries = [entry.strip() for entry in options['dictionary_entries'].split(',')]

        # Build request payload
        payload = {
            'dictionary_entries': dictionary_entries
        }

        # Add filters if provided
        filters = {}
        if options['source']:
            filters['source'] = options['source']

        if options['confidence_gt'] is not None or options['confidence_lt'] is not None:
            confidence_filter = {}
            # Validate that gt is not less than lt
            if options['confidence_gt'] is not None and options['confidence_lt'] is not None:
                if options['confidence_gt'] > options['confidence_lt']:
                    self.stdout.write(
                        self.style.ERROR('confidence_gt must be greater than confidence_lt')
                    )
                    return

            if options['confidence_gt'] is not None:
                if not (0 <= options['confidence_gt'] <= 1):
                    self.stdout.write(
                        self.style.ERROR('confidence_gt must be between 0 and 1')
                    )
                    return
                confidence_filter['greaterThan'] = options['confidence_gt']

            if options['confidence_lt'] is not None:
                if not (0 <= options['confidence_lt'] <= 1):
                    self.stdout.write(
                        self.style.ERROR('confidence_lt must be between 0 and 1')
                    )
                    return
                confidence_filter['lessThan'] = options['confidence_lt']

            filters['confidence'] = confidence_filter

        if filters:
            payload['filters'] = filters

        # Make request to the actual endpoint
        try:
            response = client.generic(
                'GET',
                reverse('definitions'),
                data=json.dumps(payload),
                content_type='application/json'
            )

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
