from django.db import models
from django.conf import settings
from pgvector.django import VectorField

class Language(models.TextChoices):
	FRENCH = 'fr'
	ARABIC = 'ar'

# Create your models here.
class Document(models.Model):
	class Filetype(models.TextChoices):
		PDF = 'pdf'

	filetype = models.CharField(
		max_length=10,
		choices=Filetype.choices,
		default=Filetype.PDF
	)

	language = models.CharField(
		max_length=2,
		choices=Language.choices,
		default=Language.FRENCH
	)

class BaseDictionaryEntry(models.Model):
	owners = models.ManyToManyField(settings.AUTH_USER_MODEL)
	word = models.CharField(max_length=100, unique=True)

	class Meta:
		abstract = True

class FrenchDictionaryEntry(BaseDictionaryEntry):
	vector = VectorField(dimensions=768)

class ArabicDictionaryEntry(BaseDictionaryEntry):
	vector = VectorField(dimensions=768)
