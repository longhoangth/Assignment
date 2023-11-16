from django.shortcuts import render, redirect
from django.db.models import Q
from .models import Room, Topic, User, Message
from .form import RoomForm, UserForm, MyUserCreationForm
from django.http import HttpResponse
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout


# Create your views here.
def loginPage(request):
	page = 'login'
	if request.user.is_authenticated:
		return redirect('home')
	if request.method == 'POST':
		username = request.POST.get('username').lower()
		password = request.POST.get('password')
		if username is None or password is None:
			messages.error(request, 'Please fill username and password')
		try:
			user = User.objects.get(username=username)
		except:
			messages.error(request, 'User does not exist')
		user = authenticate(request, username=username, password=password)
		if user is not None:
			login(request, user)
			return redirect('home')
		else:
			messages.error(request, 'Please fill the correct username and password')
	context = {'page': page}
	return render(request, 'studybud/login_register.html', context)


def registerUser(request):
	form = MyUserCreationForm()

	if request.method == 'POST':
		form = MyUserCreationForm(request.POST)
		if form.is_valid():
			user = form.save(commit=False)
			user.username = user.username.lower()
			user.save()
			login(request, user)
			return redirect('home')
		else:
			messages.error(request, 'An error occurred during registration')
	context = {'form': form}
	return render(request, 'studybud/login_register.html', context)


def logoutUser(request):
	logout(request)
	return redirect('home')


def home(request):
	req = request.GET.get('q') if request.GET.get('q') != None else ''
	rooms = Room.objects.filter(
		Q(topic__name__icontains=req) |
		Q(name__icontains=req) |
		Q(description__icontains=req)
	)
	topic = Topic.objects.all()
	room_count = rooms.count()
	room_messages = Message.objects.all().order_by('-created')
	context = {'rooms': rooms, 'topics': topic, 'room_count': room_count, 'room_messages': room_messages}
	return render(request, 'studybud/home.html', context)


def room(request, pk):
	room = Room.objects.get(pk=pk)
	room_messages = room.message_set.all().order_by('created')
	participants = room.participants.all()
	if request.method == 'POST':
		message = Message.objects.create(
			user=request.user,
			room=room,
			body=request.POST.get('body')
		)
		room.participants.add(request.user)
		return redirect('room', pk=room.id)
	context = {'room': room, 'room_messages': room_messages, 'participants': participants}
	return render(request, 'studybud/room.html', context)


def userProfile(request, pk):
	user = User.objects.get(id=pk)
	rooms = user.room_set.all()
	room_messages = user.message_set.all().order_by('created')
	topics = Topic.objects.all()
	context = {'user': user, 'rooms': rooms, 'topics': topics, 'room_messages': room_messages}
	return render(request, 'studybud/profile.html', context)


@login_required(login_url='login')
def createRoom(request):
	form = RoomForm()
	topics = Topic.objects.all()
	if request.method == 'POST':
		topic_name = request.POST.get('topic')
		topic, created = Topic.objects.get_or_create(name=topic_name)
		Room.objects.create(
			host=request.user,
			topic=topic,
			name=request.POST.get('name'),
			description=request.POST.get('description')
		)
		return redirect('home')
	context = {'form': form, 'topics': topics}
	return render(request, 'studybud/room_form.html', context)


@login_required(login_url='login')
def updateRoom(request, pk):
	room = Room.objects.get(pk=pk)
	form = RoomForm(instance=room)
	topics = Topic.objects.all()
	if request.user != room.host:
		return HttpResponse('You are not allowed to do this operation!')
	if request.method == 'POST':
		topic_name = request.POST.get('topic')
		topic, created = Topic.objects.get_or_create(name=topic_name)
		room.name = request.POST.get('name')
		room.topic = topic
		room.description = request.POST.get('description')
		room.save()
		return redirect('home')
	context = {'form': form, 'topics': topics, 'room': room}
	return render(request, 'studybud/room_form.html', context)


@login_required(login_url='login')
def deleteRoom(request, pk):
	room = Room.objects.get(pk=pk)
	if request.user != room.host:
		return HttpResponse('You are not allowed to do this operation!')
	if request.method == 'POST':
		room.delete()
		return redirect('home')
	return render(request, 'studybud/delete.html', {'obj': room})


@login_required(login_url='login')
def deleteMessage(request, pk):
	messages = Message.objects.get(id=pk)
	if request.user != messages.user:
		return HttpResponse('You are not allowed to do this operation!')
	if request.method == 'POST':
		messages.delete()
		return redirect('home')
	return render(request, 'studybud/delete.html', {'obj': messages})


@login_required(login_url='login')
def updateUser(request):
	user = request.user
	form = UserForm(instance=user)
	if request.method == 'POST':
		form = UserForm(request.POST, request.FILES, instance=user)
		if form.is_valid():
			form.save()
			return redirect('user-profile', pk=user.id)
	context = {'form': form}
	return render(request, 'studybud/update_user.html', context)
