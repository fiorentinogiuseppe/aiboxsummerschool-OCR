from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from .forms import UploadFileForm
from .process import handle_file_upload

# Create your views here.

@csrf_exempt
def main(request):
    form = UploadFileForm()
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        print('Post method reveived')
        if form.is_valid():
            redirect_name = handle_file_upload(request.FILES['text_input'])
            return HttpResponse(redirect_name)
    context = {
        'form':form
    }
    return render(request, 'postprocessing/main.html', context)

def redir(request):
    return redirect('postprocessing/')