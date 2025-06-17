from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.core.exceptions import ValidationError
import json
import magic
import spacy
from .models import (
	Language,
	Document,
	BaseDictionaryEntry,
	FrenchDictionaryEntry,
	ArabicDictionaryEntry
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

        try:
            if language_code == 'fr':
                nlp = spacy.load("fr_dep_news_trf")
            else:
                return JsonResponse({'error': 'Unsupported language for tokenization'}, status=400)
        except OSError:
            return JsonResponse({
                'error': 'Required spacy model not installed. Please install fr_dep_news_trf.'
            }, status=500)

        doc = nlp(text_content)
        processed_tokens = []

        for token in doc:
            if token.is_alpha and not token.is_stop:
                word = token.lemma_.lower()
                DictionaryEntryModel: type[BaseDictionaryEntry] = None
                if language_code == 'fr':
                    DictionaryEntryModel = FrenchDictionaryEntry
                else:
                    continue

                try:
                    dict_entry = DictionaryEntryModel.objects.get(word=word)
                except DictionaryEntryModel.DoesNotExist:
                    dict_entry = DictionaryEntryModel.objects.create(
                        word=word,
                        definition=f"Definition for {word}",
                        vector=[0.0] * 768
                    )

                if request.user not in dict_entry.owners.all():
                    dict_entry.owners.add(request.user)

                processed_tokens.append({
                    'word': word,
                    'original_text': token.text,
                    'pos': token.pos_,
                    'is_new_entry': dict_entry.pk is not None
                })

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
