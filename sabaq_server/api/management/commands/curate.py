import os
import tempfile
import subprocess
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from api.models import Definition, CurationStatus, Language

"""
NOTE: This file is full of garbage code.
It is really only a stopgap until we get a nice web interface.
If it ever becomes a real, useful part of the application, we should rewrite it.
"""

class Command(BaseCommand):
    help = '''Curate dictionary definitions by editing them in batches using your $EDITOR

    This command allows human curators to manually edit dictionary definitions.
    For each definition, you'll be dropped into your $EDITOR to make changes.

    Examples:
      python manage.py curate --language fr --n 10
      python manage.py curate --language ar --status pending
      python manage.py curate --language fr --status approved --n 5
      python manage.py curate --language fr --source wolf_wordnet --n 20
      python manage.py curate --language fr --skip-words "bonjour,merci,au revoir"
    '''

    def add_arguments(self, parser):
        """Add command line arguments"""
        parser.add_argument(
            '--language',
            type=str,
            required=True,
            choices=[choice[0] for choice in Language.choices],
            help='Language of definitions to curate (fr, ar)'
        )
        parser.add_argument(
            '--n',
            type=int,
            default=None,
            help='Number of definitions to curate (if not specified, continues until no more definitions)'
        )
        parser.add_argument(
            '--status',
            type=str,
            default=CurationStatus.PENDING,
            choices=[choice[0] for choice in CurationStatus.choices],
            help='Curation status to filter by (default: pending)'
        )
        parser.add_argument(
            '--source',
            type=str,
            default=None,
            help='Filter definitions by source (e.g., "wolf_wordnet", "wiktionary")'
        )
        parser.add_argument(
            '--skip-words',
            type=str,
            default=None,
            help='Comma-separated list of dictionary entry words to skip (e.g., "word1,word2,word3")'
        )

    def handle(self, *args, **options):
        # Get editor from environment
        editor = os.environ.get('EDITOR', 'vim')

        # Build queryset
        queryset = Definition.objects.filter(
            dictionary_entry__language=options['language'],
            curation_status=options['status']
        ).select_related('dictionary_entry')

        # Apply source filter if specified
        if options['source']:
            queryset = queryset.filter(source=options['source'])

        # Apply skip-words filter if specified
        if options['skip_words']:
            skip_words = [word.strip() for word in options['skip_words'].split(',')]
            queryset = queryset.exclude(dictionary_entry__word__in=skip_words)

        # Apply limit if specified
        if options['n'] is not None:
            queryset = queryset[:options['n']]

        total_definitions = queryset.count()
        if total_definitions == 0:
            self.stdout.write(
                self.style.WARNING('No definitions found matching the criteria, aborting')
            )
            return

        self.stdout.write(
            self.style.SUCCESS(f'Found {total_definitions} definitions to curate')
        )

        processed = 0
        for definition in queryset:
            processed += 1
            self.stdout.write(
                self.style.HTTP_INFO(f'Processing definition {processed}/{total_definitions}')
            )

            # AIDEV-NOTE: Main editing loop - handles retries and user choices
            while True:
                try:
                    success = self._edit_definition(definition, editor)
                    if success:
                        break
                    else:
                        # User chose to skip
                        self.stdout.write(
                            self.style.WARNING('Skipping definition')
                        )
                        break
                except KeyboardInterrupt:
                    self.stdout.write(
                        self.style.ERROR('\nInterrupted by user')
                    )
                    return
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f'Error editing definition: {str(e)}')
                    )

                    # Give user options
                    choice = self._get_user_choice()
                    if choice == 'try_again':
                        continue
                    elif choice == 'skip':
                        break
                    elif choice == 'quit':
                        return

        self.stdout.write(
            self.style.SUCCESS(f'Finished processing {processed} definitions')
        )

    def _edit_definition(self, definition, editor):
        """
        AIDEV-NOTE: Opens editor for a single definition and handles parsing/saving
        Returns True if saved successfully, False if user chose to skip
        """
        # Create temporary file with definition content
        content = self._format_definition_for_editing(definition)

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.sabaqcurate', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Open editor
            subprocess.run([editor, tmp_file_path], check=True)

            # Read back the edited content
            with open(tmp_file_path, 'r') as f:
                edited_content = f.read()

            # Parse the edited content
            parsed_data = self._parse_edited_content(edited_content)

            # Save to database
            self._save_definition(definition, parsed_data)

            return True

        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)

    def _format_definition_for_editing(self, definition):
        """Format definition data for editing in the editor"""
        # Get other definitions for the same dictionary entry
        other_definitions = Definition.objects.filter(
            dictionary_entry=definition.dictionary_entry
        ).exclude(id=definition.id).select_related('dictionary_entry')

        other_defs_text = ""
        if other_definitions.exists():
            other_defs_text = "Other Definitions:\n"
            for i, other_def in enumerate(other_definitions, 1):
                if other_def == definition:
                    continue
                status_display = other_def.get_curation_status_display() if hasattr(other_def, 'get_curation_status_display') else other_def.curation_status
                other_defs_text += f"  {i}. {other_def.source} (confidence: {other_def.confidence}, status: {status_display}): {other_def.text or '(no text)'}\n"
        else:
            other_defs_text = "Other Definitions: None\n"

        return f"""Dictionary Entry: {definition.dictionary_entry.word}
Source: {definition.source}
{other_defs_text}## EDIT BELOW HERE, CHANGES TO SOURCE OR DICTIONARY ENTRY WILL BE IGNORED ##
Confidence: {definition.confidence}
Status: {definition.curation_status}
Text: \"\"\"
{definition.text or ''}
\"\"\"
"""

    def _parse_edited_content(self, content):
        """
        AIDEV-NOTE: Parse the edited content and extract the editable fields
        Enforces that read-only fields are only before divider and writeable fields are only after
        """
        lines = content.split('\n')

        # Find the divider line
        divider_line = None
        for i, line in enumerate(lines):
            if line.strip().startswith('## EDIT BELOW HERE'):
                divider_line = i
                break

        if divider_line is None:
            raise ValueError("Required divider line not found")

        # Split content into before/after divider
        before_divider = lines[:divider_line]
        after_divider = lines[divider_line + 1:]

        # Check that read-only fields are only before divider
        for line in after_divider:
            if (line.strip().startswith('Dictionary Entry:') or
                line.strip().startswith('Source:') or
                line.strip().startswith('Other Definitions:')):
                raise ValueError("Read-only fields (Dictionary Entry, Source, Other Definitions) cannot be modified after divider")

        # Check that writeable fields are only after divider
        for line in before_divider:
            if (line.strip().startswith('Confidence:') or
                line.strip().startswith('Status:') or
                line.strip().startswith('Text:')):
                raise ValueError("Writeable fields (Confidence, Status, Text) must be after divider")

        data = {}

        # Parse confidence from after divider
        for line in after_divider:
            if line.startswith('Confidence:'):
                try:
                    confidence_str = line.split('Confidence:', 1)[1].strip()
                    data['confidence'] = float(confidence_str)
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid confidence value: {line}")
                break

        # Parse status from after divider
        for line in after_divider:
            if line.startswith('Status:'):
                try:
                    status = line.split('Status:', 1)[1].strip()
                    if status not in [choice[0] for choice in CurationStatus.choices]:
                        raise ValueError(f"Invalid status: {status}")
                    data['curation_status'] = status
                except IndexError:
                    raise ValueError(f"Invalid status line: {line}")
                break

        # Parse text (everything between triple quotes) from after divider
        in_text_block = False
        text_lines = []

        for line in after_divider:
            if line.strip() == 'Text: """':
                in_text_block = True
                continue
            elif line.strip() == '"""' and in_text_block:
                break
            elif in_text_block:
                text_lines.append(line)

        data['text'] = '\n'.join(text_lines)

        return data

    def _save_definition(self, definition, parsed_data):
        """Save the parsed data to the definition"""
        with transaction.atomic():
            if 'curation_status' in parsed_data:
                definition.curation_status = parsed_data['curation_status']
            if 'text' in parsed_data:
                definition.text = parsed_data['text']

            definition.save()

    def _get_user_choice(self):
        """Get user choice when an error occurs"""
        while True:
            self.stdout.write(
                self.style.WARNING('What would you like to do?')
            )
            self.stdout.write('1. Try again (edit same definition)')
            self.stdout.write('2. Skip (move to next definition)')
            self.stdout.write('3. Quit (stop editing)')

            try:
                choice = input('Enter your choice (1-3): ').strip()

                if choice == '1':
                    return 'try_again'
                elif choice == '2':
                    return 'skip'
                elif choice == '3':
                    return 'quit'
                else:
                    self.stdout.write(
                        self.style.ERROR('Invalid choice. Please enter 1, 2, or 3.')
                    )
            except (EOFError, KeyboardInterrupt):
                return 'quit'
