from django import forms

class UploadFileForm(forms.Form):
    text_input = forms.FileField()