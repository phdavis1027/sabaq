from django.urls import path
from django.contrib.auth.views import LogoutView

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("dictionary_entries", views.dictionary_entries, name="dictionary_entries"),
    path("definitions", views.definitions, name="definitions"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("logout/", LogoutView.as_view(next_page='index'), name="logout"),
    path("document", views.upload_document, name="upload_document"),
    path("export-anki", views.export_definition_set_to_anki, name="export_definition_set_to_anki")
]
