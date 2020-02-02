from django.db import models

# Create your models here.

class TextInput(models.Model):
    text_input = models.TextField(max_length=10)
    text_output = models.TextField(blank=True)