from django.http import HttpResponse
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
)


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
        if request.content_type.startswith('multipart/form-data'):
            filetype = request.POST.get('filetype')
            language_code = request.POST.get('language')
            uploaded_file = request.FILES.get('file')
        else:
            try:
                data = json.loads(request.body)
                filetype = data.get('filetype')
                language_code = data.get('language')
                uploaded_file = None
            except json.JSONDecodeError:
                return JsonResponse({'error': 'Invalid JSON data'}, status=400)

        # Validate required fields
        if not all([filetype, language_code, uploaded_file]):
            return JsonResponse({
                'error': 'Missing required fields: filetype, language, and file are required'
            }, status=400)

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

    entries = DictionaryEntry.objects.filter(**query)

    if order_bys := request.GET.get('order_bys'):
        for order_by in order_bys.split(','):
            entries = entries.order_by(order_by)

    return HttpResponse(
        serializers.serialize(
            "json",
            entries,
            fields = ["language", "word"]
        ),
        content_type="application/json"
    )
