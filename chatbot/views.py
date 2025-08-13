from django.shortcuts import render
from django.http import JsonResponse
import chatbot_response

def homepage(request):
    return render(request, 'homepage.html')
def chatbot(request):
    if request.method == 'POST':
        user_message = request.POST.get('userMessage')
        print(user_message)
        bot_response = chatbot_response.get_response(user_message)
        # Send chatbot response as JSON for the frontend to handle
        return JsonResponse({'response': bot_response})
    else:
        return render(request, 'chat.html')

def about(request):
    return render(request, 'about.html')
def contact(request):
    return render(request, 'contact.html')