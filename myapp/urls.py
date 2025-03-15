# blog/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login_view, name='login_view'),  # The home page of the blog
    path('register/',views.register,name='register'),
    path('chatbot/',views.chatbot,name='chatbot'),
    path('home/',views.home,name='home'),
    path('about/',views.about,name='about'),
    path('logout/',views.logout_view,name='logout_view'),
    path('clear_chat_history/',views.clear_chat_history,name='clear_chat_history'),
    # path('chat_history/',views.chat_history,name='chat_history')
]
