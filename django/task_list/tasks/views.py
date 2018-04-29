from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.http import Http404
from django.shortcuts import get_object_or_404
from django.views import generic
# Create your views here.
from braces.views import SelectRelatedMixin
from . import models
from . import forms
from django.contrib.auth import get_user_model
from django.contrib import messages
User = get_user_model()


class TasksList(LoginRequiredMixin, generic.ListView): #SelectRelatedMixin,
    model = models.Task
    #select_related = ('user', )


class TasksUser(LoginRequiredMixin, generic.ListView):
    model = models.Task
    template_name = "tasks/user_task_list.html"

    def get_queryset(self):
        try:
            self.task_user = User.objects.prefetch_related('tasks').get(
                username__iexact=self.kwargs.get('username'))
        except User.DoesNotExist:
            raise Http404
        else:
            return self.task_user.tasks.all()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['task_user'] = self.task_user
        return context


class TasksDetail(LoginRequiredMixin, generic.DetailView): #SelectRelatedMixin,
    model = models.Task
    #select_related = ("user", )
    #
    # def get_queryset(self):
    #     queryset = super().get_queryset()
    #     return queryset.filter(user__username__iexact=self.kwargs.get(
    #         'username'))


class TasksCreate(LoginRequiredMixin, SelectRelatedMixin, generic.CreateView):
    fields = ('title', 'message')
    model = models.Task

    def form_valid(self, form):
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.save()
        return super().form_valid(form)


class TasksDelete(LoginRequiredMixin, generic.DeleteView): #SelectRelatedMixin
    model = models.Task
    success_url = reverse_lazy('tasks:all')


    def get_queryset(self):
        queryset = super().get_queryset()
        #print(dir(self.request))
        #print(dir(self.model.pk))

        return queryset.filter()

    def delete(self, *args, **kwargs):
        messages.success(self.request, 'Task Deleted')
        print("asdas")
        return super().delete(*args, **kwargs)
