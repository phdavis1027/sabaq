from django.contrib import admin
from .models import Document, ExampleSentence, DictionaryEntry, Definition

# Register your models here.


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['id', 'filetype', 'language']
    list_filter = ['filetype', 'language']


@admin.register(ExampleSentence)
class ExampleSentenceAdmin(admin.ModelAdmin):
    list_display = ['id', 'text', 'language']
    list_filter = ['language']
    search_fields = ['text']


@admin.register(DictionaryEntry)
class DictionaryEntryAdmin(admin.ModelAdmin):
    list_display = ['id', 'word', 'language']
    list_filter = ['language']
    search_fields = ['word']
    filter_horizontal = ['owners']


@admin.register(Definition)
class DefinitionAdmin(admin.ModelAdmin):
    list_display = ['id', 'dictionary_entry', 'source', 'confidence', 'usage_count']
    list_filter = ['source', 'dictionary_entry__language']
    search_fields = ['text', 'dictionary_entry__word']
    readonly_fields = ['usage_count']
