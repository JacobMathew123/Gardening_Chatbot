from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login,authenticate,logout
from .utils import get_gardening_advice
from .models import Query
from django.contrib.auth.decorators import login_required


# Create your views here.


def home(request):
    return render(request,'pages/home.html')
def about(request):
    return render(request,'pages/about.html')

# chatbot views

# @login_required
# def chatbot(request):
#     response=None
#     chat_history=Query.objects.filter(user=request.user).order_by('-timestamp')[:10]

#     if request.method == 'POST':
#         question=request.POST.get('question')
#         response=get_gardening_advice(question)
#         Query.objects.create(user=request.user,question=question,response=response)

#     return render(request,'pages/chatbot.html',{'response':response,'chat_history':chat_history})
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import Query
from .utils import get_gardening_advice  # Import the function from utils.py

@login_required
def chatbot(request):
    response = None
    chat_history = Query.objects.filter(user=request.user).order_by('-timestamp')[:10]

    if request.method == 'POST':
        question = request.POST.get('question', '').strip()

        if question:  # Ensure the input is not empty
            try:
                response = get_gardening_advice(question)
                Query.objects.create(user=request.user, question=question, response=response)
            except Exception as e:
                response = f"An error occurred: {str(e)}"  # Handles AI errors gracefully

    return render(request, 'pages/chatbot.html', {'response': response, 'chat_history': chat_history})


# clear_history views

@login_required
def clear_chat_history(request):
    Query.objects.filter(user=request.user).delete()
    messages.success(request,'Chat history cleared')
    return redirect('chatbot')

# history views

# def chat_history(request):
#     chat_history=Query.objects.filter(user=request.user).order_by('timestamp')
#     return render(request,'chatbot.html',{'chat_history':chat_history})

# register validation

def register(request):
    if request.method =='POST':
       username=request.POST.get('username','')
       email=request.POST.get('email','')
       password=request.POST.get('password','')
       confirm_password=request.POST.get('confirm_password','')

       if not username or not email or not password:
          messages.error(request,'All fields are required')
    #    elif password != (r'/^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,}$/',password):
    #        messages.error(request,'Please enter a valid password.It must be at least 8 characters long and include at least one letter and one number')
       elif password != confirm_password:
          messages.error(request,'Password do not match')
       elif User.objects.filter(username=username).exists():
          messages.error(request,'Username already exists')
       elif User.objects.filter(email=email).exists():
          messages.error(request,'Email already exists')
       else:
          user=User.objects.create_user(username=username,email=email,password=password)
          user.save()
          messages.success(request,'Account created successfuly! Please login!')
          return redirect('login_view')
    return render(request,'pages/register.html')


# login validation

def login_view(request):
    if request.method =='POST':
        username=request.POST.get('username','')
        password=request.POST.get('password','')
        User=authenticate(request,username=username,password=password)
        if User is not None:
            login(request,User)
            return redirect('chatbot')
        else:
            messages.error(request,'Invalid username or password')
    return render(request,'pages/login.html')


# logout validation

def logout_view(request):
    logout(request)
    messages.success(request,'Logged out successfuly!')
    return redirect('home')