from djongo import models


class Recipes(models.Model):
    _id = models.ObjectIdField(auto_created=True, primary_key=True)
    title = models.CharField(max_length=256)
    category = models.CharField(max_length=256)
    imageUrl = models.CharField(max_length=256)
    duration = models.CharField(max_length=256)
    link = models.CharField(max_length=256)
    ingredients = models.JSONField(models.CharField(max_length=256),  # Define the type of elements in the array
                                   blank=True,  # Allow the array to be empty (optional)
                                   null=True,  # Allow the array to be None (optional)
                                   default=list)
    steps = models.JSONField(models.CharField(max_length=256),  # Define the type of elements in the array
                             blank=True,  # Allow the array to be empty (optional)
                             null=True,  # Allow the array to be None (optional)
                             default=list)

    def __str__(self):
        return f'{self.title} - {self.category} - {self.imageUrl} - {self.duration} - {self.ingredients} {self.link} - {self.steps}'
