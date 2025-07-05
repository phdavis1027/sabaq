from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.core.exceptions import ValidationError
from django.core import serializers
import pdb

import json
import magic
import spacy
import genanki
import tempfile
import os
from .models import (
    Language,
    Document,
    DictionaryEntry,
    Definition,
)
from .utils import json_required, file_required, random_int

# AIDEV-NOTE: Use built-in Django serializers and deserializers if at all possible
# AIDEV-NOTE: To read JSON bodies, use `request.data` and `get()` keys as you need them

def index(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if not username or not password:
            return render(request, 'api/login.html', {
                'error': 'Please provide both username and password.'
            })

        # Authenticate user
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            return render(request, 'api/login.html', {
                'error': 'Invalid username or password. Please try again.'
            })

    return render(request, 'api/login.html')


@login_required
def dashboard(request):
    return render(request, 'api/dashboard.html', {
        'user': request.user
    })


@file_required([lambda files: 'document' in files])
@json_required(['filetype', 'language'])
@csrf_exempt
@require_http_methods(["POST"])
@login_required
def upload_document(request):
    """
    AIDEV-NOTE: Authenticated endpoint for document upload and processing.
    Validates filetype using python-magic, tokenizes with spacy, and manages dictionary entries.
    Currently supports only French ('fr') language and text ('txt') files.
    """
    try:
        data = request.json
        filetype = data.get('filetype')
        language_code = data.get('language')
        uploaded_file = request.FILES.get('document')

        if language_code != 'fr':
            return JsonResponse({
                'error': 'Invalid language code. Only "fr" is currently supported.'
            }, status=400)

        if filetype != 'txt':
            return JsonResponse({
                'error': 'Invalid filetype. Only "txt" is currently supported.'
            }, status=400)

        file_content = uploaded_file.read()
        actual_filetype = magic.from_buffer(file_content, mime=True)

        if not actual_filetype.startswith('text/'):
            return JsonResponse({
                'error': f'File validation failed. Expected text file, got {actual_filetype}'
            }, status=400)

        uploaded_file.seek(0)
        text_content = file_content.decode('utf-8')

        if language_code == 'fr':
            nlp = spacy.load("fr_dep_news_trf")
        else:
            return JsonResponse({'error': 'Unsupported language for tokenization'}, status=400)

        doc = nlp(text_content)
        processed_tokens = []

        for token in doc:
            if token.is_alpha and not token.is_stop:
                word = token.lemma_.lower()

                try:
                    dict_entry = DictionaryEntry.objects.get(word=word)
                except DictionaryEntry.DoesNotExist:
                    dict_entry = DictionaryEntry.objects.create(
                        word=word,
                        definition=f"Definition for {word}",
                        language=language_code
                    )

                processed_tokens.append({
                    'word': word,
                    'original_text': token.text,
                    'pos': token.pos_,
                    'is_new_entry': dict_entry is None
                })

                if request.user not in dict_entry.owners.all():
                    dict_entry.owners.add(request.user)


        # AIDEV-NOTE: Could create Document record here if needed for tracking
        # document = Document.objects.create(
        #     filetype=Document.Filetype.TXT,
        #     language=Language.FRENCH
        # )

        return JsonResponse({
            'success': True,
            'message': f'Document processed successfully. {len(processed_tokens)} tokens processed.',
            'tokens_processed': len(processed_tokens),
            'language': language_code,
            'filetype': filetype,
            'tokens': processed_tokens[:10]
        })

    except UnicodeDecodeError:
        return JsonResponse({'error': 'File encoding error. Please ensure file is UTF-8 encoded.'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'An error occurred processing the document: {str(e)}'}, status=500)


@login_required
@require_http_methods(["GET"])
def dictionary_entries(request):
    """
    Retrieves dictionary entries owned by a particular user
    Order-Bys will be applied in the order they appear
    """
    query = {
        'owners': request.user
    }

    if langs := request.GET.get('languages'):
        query['language__in'] = langs

    entries = DictionaryEntry.objects.filter(**query).values('word', 'language')

    if order_bys := request.GET.get('order_bys'):
        for order_by in order_bys.split(','):
            entries = entries.order_by(order_by)

    data = list(entries)
    return JsonResponse(data)


@json_required(['dictionary_entries'])
@csrf_exempt
@require_http_methods(["GET"])
@login_required
def definitions(request):
    """
    AIDEV-NOTE: Authenticated endpoint for retrieving definitions based on dictionary entries and filters.
    Expects JSON body with dictionary_entries array and optional filters for source and confidence.
    Returns definitions grouped by dictionary entry word.
    """
    try:
        data = request.json

        dictionary_entries = data.get("dictionary_entries")
        if not isinstance(dictionary_entries, list):
            return JsonResponse({'error': 'dictionary_entries must be an array'}, status=400)

        if not dictionary_entries:
            return JsonResponse({'error': 'dictionary_entries cannot be empty'}, status=400)

        filters = data.get('filters', {})

        # Validate filters
        if filters:
            if 'confidence' in filters:
                confidence_filter = filters['confidence']
                if 'greaterThan' in confidence_filter:
                    gt_val = confidence_filter['greaterThan']
                    if not isinstance(gt_val, (int, float)) or not (0 <= gt_val <= 1):
                        return JsonResponse({'error': 'confidence.greaterThan must be a number between 0 and 1'}, status=400)

                if 'lessThan' in confidence_filter:
                    lt_val = confidence_filter['lessThan']
                    if not isinstance(lt_val, (int, float)) or not (0 <= lt_val <= 1):
                        return JsonResponse({'error': 'confidence.lessThan must be a number between 0 and 1'}, status=400)

        result = {}

        for entry_word in dictionary_entries:
            try:
                # Get the dictionary entry
                dict_entry = DictionaryEntry.objects.get(word=entry_word)

                # Check if user owns this entry
                if not dict_entry.owners.filter(id=request.user.id).exists():
                    result[entry_word] = []
                    continue

                # Build definition query
                definition_query = Definition.objects.filter(dictionary_entry=dict_entry)

                # Apply filters
                if filters:
                    if 'source' in filters:
                        definition_query = definition_query.filter(source=filters['source'])

                    if 'confidence' in filters:
                        confidence_filter = filters['confidence']
                        if 'greaterThan' in confidence_filter:
                            definition_query = definition_query.filter(confidence__gt=confidence_filter['greaterThan'])
                        if 'lessThan' in confidence_filter:
                            definition_query = definition_query.filter(confidence__lt=confidence_filter['lessThan'])

                # Get definitions and serialize
                definitions = definition_query.only('source', 'confidence', 'text', 'usage_count')

                result[entry_word] = serializers.serialize('json', definitions)

            except DictionaryEntry.DoesNotExist:
                # If dictionary entry doesn't exist, return empty array
                result[entry_word] = []

        return JsonResponse(result)


    except Exception as e:
        return JsonResponse({'error': f'An error occurred processing the request: {str(e)}'}, status=500)


@json_required(['deck_name', 'definition_set'])
@csrf_exempt
@require_http_methods(["POST"])
@login_required
def export_definition_set_to_anki(request):
    """
    AIDEV-NOTE: Authenticated endpoint for exporting definition sets to Anki .apkg format.
    Expects JSON body with deck_name, definition_set (from definitions endpoint), and optional sources filter.
    Returns .apkg file as binary response. Pure function - no database modifications.
    """
    try:
        data = request.json

        deck_name = data.get('deck_name')
        definition_set = data.get('definition_set')
        sources = data.get('sources', [])

        if not isinstance(definition_set, dict):
            return JsonResponse({'error': 'definition_set must be an object'}, status=400)

        if sources and not isinstance(sources, list):
            return JsonResponse({'error': 'sources must be an array'}, status=400)

        # Create Anki deck
        deck_id = random_int()
        deck = genanki.Deck(deck_id, deck_name)

        # Define note model for flashcards
        model = genanki.Model(
            random_int(),
            'Sabaq Basic Model',
            fields=[
                {'name': 'Front'},
                {'name': 'Back'},
            ],
            templates=[
                {
                    'name': 'Card 1',
                    'qfmt': '{{Front}}',
                    'afmt': '{{FrontSide}}<hr id="answer">{{Back}}',
                },
            ]
        )

        cards_created = 0

        # Process each word in the definition set
        for word, definitions_json in definition_set.items():
            if not definitions_json:
                continue

            # Parse the serialized definitions
            try:
                definitions = json.loads(definitions_json)
            except (json.JSONDecodeError, TypeError):
                #  TODO: Send to Glitchtip
                continue

            # Filter definitions by source if sources filter is provided
            if sources:
                filtered_definitions = [
                    defn for defn in definitions
                    if defn.get('fields', {}).get('source') in sources
                ]
            else:
                filtered_definitions = definitions

            if not filtered_definitions:
                continue

            # Find definition with highest confidence
            best_definition = max(
                filtered_definitions,
                key=lambda d: d.get('fields', {}).get('confidence', 0)
            )

            definition_text = best_definition.get('fields', {}).get('text', '')

            if not definition_text:
                continue

            # Create Anki note
            note = genanki.Note(
                model=model,
                fields=[word, definition_text]
            )

            deck.add_note(note)
            cards_created += 1

        if cards_created == 0:
            return JsonResponse({'error': 'No valid definitions found to export'}, status=400)

        # Generate .apkg file
        package = genanki.Package(deck)

        # Create temporary file
        # Can be cleaned up easily since we can trivially just purge all .apkg files in the directory
        with tempfile.NamedTemporaryFile(suffix='.apkg', delete=False) as tmp_file:
            package.write_to_file(tmp_file.name)
            tmp_file_path = tmp_file.name

        try:
            # Read the .apkg file and return as response
            with open(tmp_file_path, 'rb') as f:
                apkg_data = f.read()

            response = HttpResponse(
                apkg_data,
                content_type='application/octet-stream'
            )
            response['Content-Disposition'] = f'attachment; filename="{deck_name}.apkg"'

            return response

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)


    except Exception as e:
        return JsonResponse({'error': f'An error occurred exporting to Anki: {str(e)}'}, status=500)
