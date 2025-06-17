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
	definition = models.TextField()
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
		choices=[choice for choice in DefinitionSource.choices];
	)
	confidence = models.FloatField(default=0.0)

	usage_count = models.IntegerField(default=0)

	class Meta:
		constraints = [
			models.CheckConstraint(
				check=models.Q(
					models.Q(
						dictionary_entry__language='fr',
						source__in=[choice[0] for choice in FrenchDefinitionSource.choices]
					) |
					models.Q(
						dictionary_entry__language='ar',
						source__in=[choice[0] for choice in ArabicDefinitionSource.choices]
					)
				),
				name='source_matches_entry_language',
				violation_error_message='Definition source must be valid for dictionary entry language'
			)
		]

	def clean(self):
		super().clean()

		if not self.dictionary_entry or not self.source:
			return

		entry_language = self.dictionary_entry.language

		language_source_map = {
			Language.FRENCH: [choice[0] for choice in FrenchDefinitionSource.choices],
			Language.ARABIC: [choice[0] for choice in ArabicDefinitionSource.choices],
		}

		valid_sources = language_source_map.get(entry_language, [])

		if self.source not in valid_sources:
			source_class_name = {
				Language.FRENCH: 'FrenchDefinitionSource',
				Language.ARABIC: 'ArabicDefinitionSource',
			}.get(entry_language, 'Unknown')

			raise ValidationError({
				'source': f'Source "{self.source}" is not valid for {entry_language} entries. '
						f'Valid sources from {source_class_name}: {", ".join(valid_sources)}'
			})

	def save(self, *args, **kwargs):
		self.clean()
		super().save(*args, **kwargs)
