from django.core.validators import FileExtensionValidator, validate_image_file_extension
from rest_framework import serializers
from derma.models import *


class NutritionSerializer(serializers.Serializer):
    amount = serializers.DecimalField(max_digits=10, decimal_places=2, coerce_to_string=False)
    unit = serializers.CharField(max_length=256)


class NutritionInfoSerializer(serializers.Serializer):
    proteins = NutritionSerializer()
    carbohydrates = NutritionSerializer()
    fats = NutritionSerializer()
    calories = NutritionSerializer()


class IngredientSerializer(serializers.Serializer):
    quantity = serializers.CharField(max_length=256)
    unit = serializers.CharField(max_length=256)
    name = serializers.CharField(max_length=256, source='ingredient')


class ObjectIIdField(serializers.Field):
    def to_representation(self, value):
        return str(value)


class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField(validators=[FileExtensionValidator(allowed_extensions=['png', 'jpg', 'jpeg']),
                                               validate_image_file_extension])


class RecipesSerializer(serializers.ModelSerializer):
    ingredients = IngredientSerializer(many=True)
    nutritionInfo = NutritionInfoSerializer()
    steps = serializers.ListField(child=serializers.CharField())
    _id = ObjectIIdField()

    class Meta:
        model = EcuatorianRecipes
        fields = '__all__'
