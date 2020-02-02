from django import forms
from .models import TextInput

class InputTextForm(forms.ModelForm):
    class Meta:
        model = TextInput
        fields = '__all__'