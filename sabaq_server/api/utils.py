import json
from functools import wraps
import uuid
import random

from django.http import JsonResponse
from django.db import models
from django.utils import timezone


def json_required(required_fields=None):
    """
    AIDEV-NOTE: Decorator that automatically parses JSON request bodies.
    Adds request.json attribute containing parsed JSON data.
    Returns 400 error response if JSON parsing fails or required fields are missing.

    Args:
        required_fields (list, optional): List of required field names in the JSON body
    """
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            if request.content_type == 'application/json' or request.body:
                try:
                    request.json = json.loads(request.body)
                except json.JSONDecodeError:
                    return JsonResponse({'error': 'Invalid JSON data'}, status=400)
            else:
                request.json = {}

            # Check required fields
            if required_fields:
                missing_fields = []
                for field in required_fields:
                    if field not in request.json:
                        missing_fields.append(field)

                if missing_fields:
                    return JsonResponse({
                        'error': f'Missing required fields: {", ".join(missing_fields)}'
                    }, status=400)

            return view_func(request, *args, **kwargs)
        return wrapper

    # Allow decorator to be used with or without parameters
    if callable(required_fields):
        # Called as @json_required without parentheses
        view_func = required_fields
        required_fields = None
        return decorator(view_func)
    else:
        # Called as @json_required() or @json_required(['field1', 'field2'])
        return decorator


def file_required(validation_funcs=None):
    """
    AIDEV-NOTE: Decorator that validates files are attached to the request.
    Returns 400 error response if validation fails.

    Args:
        validation_func (callable, optional): Function that takes request.FILES and returns truthy/falsy.
                                            If None, just checks that any file is attached.
    """
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            if validation_funcs is None:
                # Just check that there's at least one file
                if not request.FILES:
                    return JsonResponse({
                        'error': 'No files attached to request'
                    }, status=400)
            else:
                # Use the custom validation function
                for func in validation_funcs:
                    if not func(request.FILES):
                        return JsonResponse({
                            'error': 'File validation failed'
                        }, status=400)

            return view_func(request, *args, **kwargs)
        return wrapper

    # Allow decorator to be used with or without parameters
    if callable(validation_funcs):
        # Called as @file_required without parentheses (validation_funcs is actually the view function)
        view_func = validation_funcs
        validation_funcs = None
        return decorator(view_func)
    else:
        # Called as @file_required() or @file_required(validation_funcs)
        return decorator

# Nabbed from: https://stackoverflow.com/questions/35210753/how-does-a-django-uuidfield-generate-a-uuid-in-postgresql
def random_int():
    return random.randint(0, 281474976710655)

def random_uuid():
    return uuid.uuid1(random_int())

def default_now():
    return timezone.now()

def timestamped(cls):
    """
    AIDEV-NOTE: Decorator that automatically adds created and modified timestamp fields
    to Django models and overrides the save method to update them automatically.

    Usage:
        @timestamped
        class MyModel(models.Model):
            # your fields here
            pass

    This will add:
    - created: DateTimeField(editable=False) - set only on creation
    - modified: DateTimeField() - updated on every save
    """
    if not issubclass(cls, models.Model):
        raise TypeError("@timestamped can only be applied to Django Model classes")

    # Add the timestamp fields
    cls.add_to_class('created',
        models.DateTimeField(editable=False, default=default_now))
    cls.add_to_class('modified',
        models.DateTimeField(default=default_now))

    # Store the original save method
    original_save = cls.save

    def save(self, *args, **kwargs):
        """On save, update timestamps"""
        if not self.id:
            self.created = timezone.now()
        self.modified = timezone.now()
        return original_save(self, *args, **kwargs)

    # Override the save method
    cls.save = save

    return cls
