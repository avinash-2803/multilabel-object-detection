from django.urls import path
from detector import views

urlpatterns = [
    path("",        views.index,   name="index"),
    path("analyse/", views.analyse, name="analyse"),
]
