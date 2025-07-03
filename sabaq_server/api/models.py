from django.db import models
from django.conf import settings
from django.core.exceptions import ValidationError
from pgvector.django import VectorField


class Language(models.TextChoices):
    FRENCH = 'fr'
    ARABIC = 'ar'

# Create your models here.


class Document(models.Model):
    class Filetype(models.TextChoices):
        PDF = 'pdf'
        TXT = 'txt'

    filetype = models.CharField(
        max_length=10,
        choices=Filetype.choices,
        default=Filetype.TXT
    )

    language = models.CharField(
        max_length=2,
        choices=Language.choices,
        default=Language.FRENCH
    )


class ExampleSentence(models.Model):
    text = models.TextField()
    language = models.CharField(
        max_length=2,
        choices=Language.choices,
        default=Language.FRENCH
    )


class DictionaryEntry(models.Model):
    owners = models.ManyToManyField(settings.AUTH_USER_MODEL)
    word = models.CharField(max_length=100, unique=True)
    language = models.CharField(
        max_length=2,
        choices=Language.choices,
        default=Language.FRENCH
    )


class DefinitionSource(models.TextChoices):
    pass


class FrenchDefinitionSource(DefinitionSource):
    WOLF_WORDNET = 'wolf_wordnet'
    WIKTIONARY = 'wiktionary'


class ArabicDefinitionSource(DefinitionSource):
    ARATOOLS = 'aratools'


class Definition(models.Model):
    dictionary_entry = models.ForeignKey(
        'DictionaryEntry',
        on_delete=models.CASCADE
    )
    source = models.CharField(
        max_length=32,
        choices=[choice for choice in DefinitionSource.choices]
    )
    confidence = models.FloatField(default=0.0)
    text = models.TextField(blank=False, null=True)

    usage_count = models.IntegerField(default=0)


    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)
