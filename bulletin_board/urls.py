from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='bulletin-home'),
    path('bulletin_results', views.results, name='bulletin-results')
]