from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("documents/", views.upload_document, name="upload_document"),
]
