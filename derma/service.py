import json
import math
import io

import cv2
from PIL import Image
from derma.models import *
from derma.serializers import RecipesSerializer, ImageUploadSerializer
from .apps import DelliConfig
import supervision as sv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from bson import ObjectId
import concurrent.futures
import time

class RecipeService:
    def get_recipes(self, query):
        recipes = []
        if query:
            recipes = self.get_recommendations([query])
        else:
            recipes = EcuatorianRecipes.objects.all()
        return RecipesSerializer(recipes, many=True).data

    def predict_ingredients(self, image,index):
        image_upload_serializer = ImageUploadSerializer(data={'image': image})
        if not image_upload_serializer.is_valid():
            return
        image_files = image_upload_serializer.validated_data['image']
        image_files.seek(0)
        image_bytes = image_files.read()
        img = Image.open(io.BytesIO(image_bytes))
        return DelliConfig.ingredients_models[index](img, save=False, conf=0.60)[0]
    def predict_ingredients_image(self,serializer,index):
        result = self.predict_ingredients(serializer,index)
        detections = sv.Detections.from_ultralytics(result)
        image = serializer.validated_data['image']
        image.seek(0)
        image_array = np.asarray(bytearray(image.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        annotated_image = sv.BoxAnnotator().annotate(scene=image, detections=detections)
        annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)
        annotated_image = cv2.resize(annotated_image, (640, 640))
        return annotated_image
    def get_ingredients(self, serializers: [ImageUploadSerializer]):
        annotated_images = []
        with concurrent.futures.ThreadPoolExecutor(6) as executor:
            futures = []
            for index,serializer in enumerate(serializers):
                futures.append(executor.submit(self.predict_ingredients_image, serializer,index))
            for future in concurrent.futures.as_completed(futures):
                annotated_images.append(future.result())
                # detections = sv.Detections.from_ultralytics(future.result())
                # image = serializer.validated_data['image']
                # image.seek(0)
                # image_array = np.asarray(bytearray(image.read()), dtype=np.uint8)
                # image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                # annotated_image = sv.BoxAnnotator().annotate(scene=image, detections=detections)
                # annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)
                # annotated_image = cv2.resize(annotated_image, (640, 640))
                # annotated_images.append(annotated_image)

        # for serializer in serializers:
        #     result = self.predict_ingredients(serializer)
        #     detections = sv.Detections.from_ultralytics(result)
        #     image = serializer.validated_data['image']
        #     image.seek(0)
        #     image_array = np.asarray(bytearray(image.read()), dtype=np.uint8)
        #     image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        #     annotated_image = sv.BoxAnnotator().annotate(scene=image, detections=detections)
        #     annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)
        #     annotated_image = cv2.resize(annotated_image, (640, 640))
        #     annotated_images.append(annotated_image)
        num_images = len(annotated_images)
        grid_cols = math.ceil(math.sqrt(num_images))
        grid_rows = math.ceil(num_images / grid_cols)
        while len(annotated_images) < grid_rows * grid_cols:
            blank_image = np.zeros((640, 640, 3), dtype=np.uint8)
            annotated_images.append(blank_image)
        rows = [
            np.hstack(annotated_images[i * grid_cols:(i + 1) * grid_cols])
            for i in range(grid_rows)
        ]
        grid_image = np.vstack(rows)
        _, buffer = cv2.imencode('.jpg', grid_image)
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
        recipes = list(collection_recipes.aggregate(aggregate_pipeline))
        return recipes

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
        recipes_id = list(
            map(lambda x: ObjectId(x), DelliConfig.recipes_df.loc[cluster_recipes.index].head()['_id'].values))
        print(recipes_id)
        return list(collection_recipes.find({'_id': {"$in": recipes_id}}))

    def get_recipes_by_cluster(self, cluster):
        return DelliConfig.recipes_df[DelliConfig.recipes_df['cluster'] == cluster][
            DelliConfig.ingredient_classes.values()]

    def one_hot_encoding(self, detected_ingredients):
        one_hot = {ingredient: 0 for ingredient in DelliConfig.ingredient_classes.values()}
        for ingredient in detected_ingredients:
            if ingredient in one_hot:
                one_hot[ingredient] = 1
        return one_hot
    def get_precict_recipes(self, images):
        total_ingredients = []
        print("entro al metodoasdjfadslf")
        start = time.time()
        images = images[:6]
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            for index,image in enumerate(images):
                print(f"size: {image.size}")
                futures.append(executor.submit(self.predict_ingredients, image,index))
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                detected_ingredients = list(set(DelliConfig.ingredient_classes[int(cls)] for cls in result.boxes.cls))
                total_ingredients += detected_ingredients
    # def get_precict_recipes(self, serializers: [ImageUploadSerializer]):
    #     total_ingredients = []
    #     print("entro al metodoasdjfadslf")
    #     start = time.time()
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    #         futures = []
    #         for index,serializer in enumerate(serializers):
    #             futures.append(executor.submit(self.predict_ingredients, serializer,index))
    #         for future in concurrent.futures.as_completed(futures):
    #             result = future.result()
    #             detected_ingredients = list(set(DelliConfig.ingredient_classes[int(cls)] for cls in result.boxes.cls))
    #             total_ingredients += detected_ingredients

        # for serializer in serializers:
        #     result = self.predict_ingredients(serializer,0)
        #     detected_ingredients = list(set(DelliConfig.ingredient_classes[int(cls)] for cls in result.boxes.cls))
        #     total_ingredients += detected_ingredients
        end = time.time()
        print(f"tiempo de ejecucion {end - start}")
        if len(total_ingredients) == 0:
            return []
        # return RecipesSerializer(self.get_recipes_by_detected_ingredients_model(detected_ingredients), many=True).data
        return RecipesSerializer(self.get_recommendations(total_ingredients), many=True).data

    def get_recommendations(self, ingredients: []):
        results = DelliConfig.vector_store.similarity_search_with_score(" ".join(ingredients), k=5)
        recommended_recipes = []
        for res, score in results:
            recommended_recipes.append(json.loads(res.page_content))
        return recommended_recipes

        # original_recipes = self.get_recipes_by_detected_ingredients_query(ingredients)
        # for r in original_recipes:
        #     r['_id'] = str(r['_id'])
        # prompt_recipes = copy.deepcopy(original_recipes)
        # for recipe in prompt_recipes:
        #     del recipe['title']
        #     del recipe['link']
        #     del recipe['imageUrl']
        # model = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")
        # response = model.invoke(self.get_prompt(ingredients, prompt_recipes))
        # print(response)
        # regex = '\[(.*?)\]'
        # recipes_ids = re.search(regex, response.content).group(1)
        # recipes_ids = list(map(lambda id: id.strip(), recipes_ids.split(',')))
        # print(recipes_ids)
        # filtered_recipes = list(filter(lambda r: r['_id'] in recipes_ids, original_recipes))
        # sortered_recipes = sorted(filtered_recipes, key=lambda r: recipes_ids.index(r['_id']))
        # print(list(map(lambda r: r['_id'],sortered_recipes)))
        # return sortered_recipes

    def get_prompt(self, ingredientes: [], recetas_ecuatorianas_relevantes: [], num_recomendaciones=10):
        prompt = f"""
            Eres un chef experto en cocina ecuatoriana. El usuario tiene los siguientes ingredientes:
            {', '.join(ingredientes)}

            Analiza las siguientes recetas y recomienda las {num_recomendaciones} mejores opciones que se 
            pueden preparar con estos ingredientes.  Considera la cantidad de ingredientes disponibles, 
            la duración de la preparación, los pasos y la categoría de la receta.  
            
            Recetsa a considerar:
            {recetas_ecuatorianas_relevantes}
            Devuelve únicamente una lista de IDs de las {num_recomendaciones} mejores recetas, 
            separadas por comas y encerradas entre corchetes.  
            Ejemplo de respuesta: [1, 5, 8]
            """
        return prompt


if __name__ == '__main__':
    service = RecipeService()
    service.recommendation(['limon','pollo'])
