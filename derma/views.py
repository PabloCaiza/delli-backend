import time

from django.http import HttpResponse
from rest_framework import status
from .service import *
from rest_framework.views import APIView
from rest_framework.request import Request
from rest_framework.response import Response
from .serializers import *



def get_images(request):
    images = request.FILES.getlist('image')
    if not images:
        return Response({'error': 'No image'}, status=status.HTTP_400_BAD_REQUEST)
    result_images = []
    for image in images:
        image_upload_serializer = ImageUploadSerializer(data={'image': image})
        if not image_upload_serializer.is_valid():
            continue
        result_images.append(image_upload_serializer)
    return result_images[:6]
class RecipeView(APIView):

    def __init__(self):
        self.service = RecipeService()
    def get(self, request, format=None):
        query = request.GET.get('query', '')
        return Response(self.service.get_recipes(query), status=status.HTTP_200_OK)

    def post(self, request: Request):
        start = time.time()
        print("entro al metodo post")
        images = request.FILES.getlist('image')
        final = time.time()
        print(f"tiempo total {final - start}")
        if not images:
            return Response({'error': 'No image'}, status=status.HTTP_400_BAD_REQUEST)
        # result_images = get_images(request)


        # if not result_images:
        #     return Response({'error':'only image format supported, .jpg , .png, .jpeg'}, status=status.HTTP_400_BAD_REQUEST)
        return Response(self.service.get_precict_recipes(images), status=status.HTTP_200_OK)


class IngredientView(APIView):
    def __init__(self):
        self.service = RecipeService()
    def post(self, request: Request):
        result_images = get_images(request)
        if not result_images:
            return Response({'error': 'only image format supported, .jpg , .png, .jpeg'},
                            status=status.HTTP_400_BAD_REQUEST)
        image_buffer = self.service.get_ingredients(result_images)
        return HttpResponse(image_buffer.tobytes(),content_type="image/jpeg")