from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse
import json
import os

User = get_user_model()


class Command(BaseCommand):
    help = '''Test the export-anki endpoint by providing a JSON file with definition set

    This command provides a transparent CLI interface to test the export-anki API endpoint
    for desk checking purposes by making actual HTTP requests.

    The definition set JSON file should contain the same format as returned by the definitions endpoint:
    {
      "word1": "[{\"model\": \"api.definition\", \"pk\": 1, \"fields\": {\"source\": \"wordnet\", \"confidence\": 0.9, \"text\": \"Definition text\", \"usage_count\": 15}}]",
      "word2": "[{\"model\": \"api.definition\", \"pk\": 2, \"fields\": {\"source\": \"manual\", \"confidence\": 0.85, \"text\": \"Another definition\", \"usage_count\": 8}}]"
    }

    Examples:
      python manage.py export_anki_query --user 1 --deck_name "My French Deck" --definition_set /path/to/definitions.json
      python manage.py export_anki_query --user 2 --deck_name "Test Deck" --definition_set ./definitions.json --sources wordnet,manual
      python manage.py export_anki_query --user 3 --deck_name "Advanced French" --definition_set definitions.json --output my_deck.apkg
    '''

    def add_arguments(self, parser):
        """Add command line arguments"""
        parser.add_argument(
            '--user',
            type=int,
            required=True,
            help='User ID to export Anki deck for'
        )
        parser.add_argument(
            '--deck_name',
            type=str,
            required=True,
            help='Name for the Anki deck'
        )
        parser.add_argument(
            '--definition_set',
            type=str,
            required=True,
            help='Path to JSON file containing definition set data (same format as definitions endpoint output)'
        )
        parser.add_argument(
            '--sources',
            type=str,
            default=None,
            help='Comma-separated list of sources to filter by (e.g., "wordnet,manual")'
        )
        parser.add_argument(
            '--output',
            type=str,
            default=None,
            help='Output filename for the .apkg file (default: uses deck_name)'
        )

    def handle(self, *args, **options):
        try:
            user = User.objects.get(id=options['user'])
        except User.DoesNotExist:
            self.stdout.write(
                self.style.ERROR(f'User with ID {options["user"]} does not exist')
            )
            return

        # Read and parse the definition set JSON file
        definition_set_path = options['definition_set']
        if not os.path.exists(definition_set_path):
            self.stdout.write(
                self.style.ERROR(f'Definition set file not found: {definition_set_path}')
            )
            return

        try:
            with open(definition_set_path, 'r', encoding='utf-8') as f:
                definition_set = json.load(f)
        except json.JSONDecodeError as e:
            self.stdout.write(
                self.style.ERROR(f'Invalid JSON in definition set file: {str(e)}')
            )
            return
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error reading definition set file: {str(e)}')
            )
            return

        # Create test client and authenticate user
        client = Client()
        client.force_login(user)

        # Build request payload
        payload = {
            'deck_name': options['deck_name'],
            'definition_set': definition_set
        }

        # Add sources filter if provided
        if options['sources']:
            sources = [source.strip() for source in options['sources'].split(',')]
            payload['sources'] = sources
            self.stdout.write(f'Filtering by sources: {sources}')

        # Make request to the actual endpoint
        try:
            response = client.generic(
                'POST',
                reverse('export_definition_set_to_anki'),
                data=json.dumps(payload),
                content_type='application/json'
            )

            if response.status_code == 200:
                # Save the .apkg file
                output_filename = options['output'] or f"{options['deck_name']}.apkg"

                with open(output_filename, 'wb') as f:
                    f.write(response.content)

                self.stdout.write(
                    self.style.SUCCESS(f'Anki deck exported successfully to: {output_filename}')
                )

            else:
                self.stdout.write(
                    self.style.ERROR(f'Request failed with status {response.status_code}')
                )
                try:
                    error_data = response.json()
                    self.stdout.write(json.dumps(error_data, indent=2))
                except:
                    self.stdout.write(response.content.decode())

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error making request: {str(e)}')
            )
