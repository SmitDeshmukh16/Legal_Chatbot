from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),  # Homepage route
    path('chat/', views.chatbot, name='chatbot'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
]
