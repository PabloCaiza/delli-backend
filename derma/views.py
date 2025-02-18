from django.http import HttpResponse
from rest_framework import status
from .service import *
from rest_framework.views import APIView
from rest_framework.request import Request
from rest_framework.response import Response
from .serializers import *


# from derma.serializers import LesionRequestSerializer


class RecipeView(APIView):

    def __init__(self):
        self.service = RecipeService()
    def get(self, request, format=None):
        return Response(self.service.get_recipes(), status=status.HTTP_200_OK)

    def post(self, request: Request):
        image = request.FILES.get('image')
        if not image:
            return Response({'error': 'No image'}, status=status.HTTP_400_BAD_REQUEST)
        print(image)
        image_upload_serializer = ImageUploadSerializer(data={'image': image})
        print(image_upload_serializer.is_valid())
        if not image_upload_serializer.is_valid():
            return Response({'error':'only image format supported, .jpg , .png, .jpeg'}, status=status.HTTP_400_BAD_REQUEST)
        return Response(self.service.get_precict_recipes(image_upload_serializer), status=status.HTTP_200_OK)


class IngredientView(APIView):
    def __init__(self):
        self.service = RecipeService()
    def post(self, request: Request):
        image = request.FILES.get('image')
        if not image:
            return Response({'error': 'No image'}, status=status.HTTP_400_BAD_REQUEST)
        image_upload_serializer = ImageUploadSerializer(data={'image': image})
        if not image_upload_serializer.is_valid():
            return Response({'error': 'only image format supported, .jpg , .png, .jpeg'},
                            status=status.HTTP_400_BAD_REQUEST)
        image_buffer = self.service.get_ingredients(image_upload_serializer)
        return HttpResponse(image_buffer.tobytes(),content_type="image/jpeg")