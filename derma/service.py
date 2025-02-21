import cv2
from PIL import Image
from derma.models import *
from derma.serializers import RecipesSerializer
from .apps import DelliConfig
import supervision as sv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from bson import ObjectId


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

    def get_recipes_by_detected_ingredients_query(self, detected_ingredients):
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

    def get_recipes_by_detected_ingredients_model(self, detected_ingredients):
        encoding_detected_ingredients = self.one_hot_encoding(detected_ingredients)
        ingredients_df = pd.DataFrame([encoding_detected_ingredients])
        predicted_cluster = DelliConfig.recommendation_model.predict(ingredients_df)[0]
        similar_recipes = self.get_recipes_by_cluster(predicted_cluster)
        ingredients_df_vectors = ingredients_df.values
        similar_recipes_vectors = similar_recipes.values
        similarities = cosine_similarity(ingredients_df_vectors, similar_recipes_vectors)[0]
        cluster_recipes = similar_recipes.copy()
        cluster_recipes['similarity'] = similarities
        cluster_recipes = cluster_recipes.sort_values(by='similarity', ascending=False)
        collection_recipes = DelliConfig.mongo_db['derma_recipes']
        recipes_id = list(map(lambda x: ObjectId(x), DelliConfig.recipes_df.loc[cluster_recipes.index].head()['_id'].values))
        print(recipes_id)
        return list(collection_recipes.find({'_id': {"$in":recipes_id}}))

    def get_recipes_by_cluster(self, cluster):
        return DelliConfig.recipes_df[DelliConfig.recipes_df['cluster'] == cluster][
            DelliConfig.ingredient_classes.values()]

    def one_hot_encoding(self, detected_ingredients):
        one_hot = {ingredient: 0 for ingredient in DelliConfig.ingredient_classes.values()}
        for ingredient in detected_ingredients:
            if ingredient in one_hot:
                one_hot[ingredient] = 1
        return one_hot

    def get_precict_recipes(self, serializer):
        result = self.predict_ingredients(serializer)
        detected_ingredients = list(set(DelliConfig.ingredient_classes[int(cls)] for cls in result.boxes.cls))
        if len(detected_ingredients) == 0:
            return []
        return RecipesSerializer(self.get_recipes_by_detected_ingredients_model(detected_ingredients), many=True).data
