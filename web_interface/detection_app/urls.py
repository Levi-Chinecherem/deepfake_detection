# detection_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_video, name='upload'),
    path('results/<str:video_name>/', views.get_results, name='results'),
]