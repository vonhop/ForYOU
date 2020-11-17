from django.urls import path
from . import views

app_name = 'nlp'
urlpatterns = [
    # ex: /polls/
    path('', views.index, name='index'),
    path('imagepoem/', views.image_poem, name='image_poem')
    # # ex: /polls/5/
    # path('<int:question_id>/', views.detail, name='detail'),
    # # ex: /polls/5/results/
    # path('<int:question_id>/results/', views.results, name='results'),
    # # ex: /polls/5/vote/
    # path('<int:question_id>/vote/', views.vote, name='vote'),
]
