from djongo import models


class Nutrition(models.Model):
    amount = models.DecimalField()
    unit = models.CharField(max_length=256)

    class Meta:
        abstract = True


class NutritionInfo(models.Model):
    proteins = models.EmbeddedField(model_container=Nutrition)
    carbohydrates = models.EmbeddedField(model_container=Nutrition)
    fats = models.EmbeddedField(model_container=Nutrition)
    calories = models.EmbeddedField(model_container=Nutrition)

    class Meta:
        abstract = True


class Ingredient(models.Model):
    quantity = models.CharField(max_length=256)
    unit = models.CharField(max_length=256)
    ingredient = models.CharField(max_length=256)

    class Meta:
        abstract = True


class EcuatorianRecipes(models.Model):
    _id = models.ObjectIdField(auto_created=True, primary_key=True)
    title = models.CharField(max_length=256)
    category = models.CharField(max_length=256)
    imageUrl = models.CharField(max_length=256)
    duration = models.CharField(max_length=256)
    link = models.CharField(max_length=256)
    portions = models.CharField(max_length=256)
    ingredients = models.ArrayField(model_container=Ingredient)
    steps = models.JSONField(models.CharField(max_length=256),  # Define the type of elements in the array
                             blank=True,  # Allow the array to be empty (optional)
                             null=True,  # Allow the array to be None (optional)
                             default=list)
    nutritionInfo = models.EmbeddedField(model_container=NutritionInfo)

def __str__(self):
    return f'{self.title} - {self.category} - {self.imageUrl} - {self.duration} - {self.ingredients} {self.link} - {self.steps}'
