import cv2
from PIL import Image
from derma.models import *
from derma.serializers import RecipesSerializer
from .apps import DelliConfig
import supervision as sv
import numpy as np


class RecipeService:
    def get_recipes(self):
        recipes = Recipes.objects.all()
        return RecipesSerializer(recipes, many=True).data

    def predict_ingredients(self, serializer):
        img = Image.open(serializer.validated_data['image'])
        return DelliConfig.ingredientModel(img, save=False)[0]

    def get_ingredients(self, serializer):
        result = self.predict_ingredients(serializer)
        detections = sv.Detections.from_ultralytics(result)
        image = serializer.validated_data['image']
        image.seek(0)
        image_array = np.asarray(bytearray(image.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        annotated_image = sv.BoxAnnotator().annotate(scene=image, detections=detections)
        annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)
        _, buffer = cv2.imencode('.jpg', annotated_image)
        return buffer

    def get_recipes_by_detected_ingredients(self, detected_ingredients):
        aggregate_pipeline = [{'$match':
                                   {'ingredients':
                                        {'$in': detected_ingredients}
                                    }
                               },
                              {'$addFields':
                                   {'matchingIngredients':
                                        {'$size':
                                             {'$setIntersection': ['$ingredients', detected_ingredients]}
                                         }
                                    }
                               },
                              {'$sort':
                                   {'matchingIngredients': -1}
                               }]
        collection_recipes = DelliConfig.mongo_db['derma_recipes']
        return list(collection_recipes.aggregate(aggregate_pipeline))

    def get_precict_recipes(self, serializer):
        result = self.predict_ingredients(serializer)
        detected_ingredients = list(set(DelliConfig.ingredient_classes[int(cls)] for cls in result.boxes.cls))
        return RecipesSerializer(self.get_recipes_by_detected_ingredients(detected_ingredients), many=True).data
