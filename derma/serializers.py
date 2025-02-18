from django.core.validators import FileExtensionValidator, validate_image_file_extension
from rest_framework import serializers
from derma.models import Recipes



class ObjectIIdField(serializers.Field):
    def to_representation(self, value):
        return str(value)
class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField(validators=[FileExtensionValidator(allowed_extensions=['png', 'jpg', 'jpeg']),
                                               validate_image_file_extension])


class RecipesSerializer(serializers.ModelSerializer):
    ingredients = serializers.ListField(child=serializers.CharField())
    steps = serializers.ListField(child=serializers.CharField())
    _id = ObjectIIdField()

    class Meta:
        model = Recipes
        fields = '__all__'
