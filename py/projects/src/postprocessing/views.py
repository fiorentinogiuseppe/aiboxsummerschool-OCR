from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from .forms import InputTextForm
from .process import handle_text_upload

# Create your views here.

@csrf_exempt
def main(request):
    form = InputTextForm()
    if request.method == 'POST':
        form = InputTextForm(request.POST)
        print('Post method reveived')
        if form.is_valid():
            received_text = request.POST['text_input']
            altered_text = handle_text_upload(received_text)
            form = InputTextForm(initial={'text_input': received_text, 'text_output': altered_text})
    context = {
        'form':form
    }
    return render(request, 'postprocessing/main.html', context)

def redir(request):
    return redirect('postprocessing/')