from django.db import models

# Create your models here.
class Documents(models.Model):
    doc = models.FileField()