from django.urls import path
from django.contrib.auth.views import LogoutView

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("logout/", LogoutView.as_view(next_page='index'), name="logout"),
    path("document", views.upload_document, name="upload_document"),
]
