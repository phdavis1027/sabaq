from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.core.exceptions import ValidationError
from django.core import serializers

import json
import magic
import spacy
from .models import (
    Language,
    Document,
    DictionaryEntry,
    Definition,
)
from .utils import json_required, file_required

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

        # pdb.set_trace()
        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'An error occurred processing the request: {str(e)}'}, status=500)
