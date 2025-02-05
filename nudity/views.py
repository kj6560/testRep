from django.http import HttpResponse
import joblib
def home(request):
    return HttpResponse("Hello, Django is working!")
