import json
import os

from django.apps import AppConfig
from ultralytics import YOLO
from dotenv import load_dotenv
from pymongo import *
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
import joblib
load_dotenv()


class DelliConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'derma'
    ingredientModel = YOLO('models/best.pt')
    ingredient_classes = {0: 'ajo', 1: 'cebolla paiteña', 2: 'huevo', 3: 'leche', 4: 'limón',
                          5: 'papa', 6: 'plátano verde', 7: 'pollo', 8: 'queso', 9: 'tomate'}
    ingredients_models = [YOLO('models/best.pt') for _ in range(6)]
    mongo_db = MongoClient(os.getenv('MONGO_HOST'))['delli']
    recipes_df = pd.read_csv('models/recipes.csv')
    recommendation_model = joblib.load('models/recomendation_model.pkl')
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = {}
    vector_store = MongoDBAtlasVectorSearch(
        collection = mongo_db["ecuatorian_recipes_vector"],
        embedding = embeddings,
        index_name = "langchain-test-index-vectorstores",
        relevance_score_fn = "cosine"
    )
