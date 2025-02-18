from django.urls import path
from derma.views import RecipeView, IngredientView

urlpatterns = [
    path('recipes', RecipeView.as_view()),
    path('ingredients', IngredientView.as_view())
]
