from django.urls import path
from tasks import views as task_views

app_name = "tasks"

urlpatterns = [
    path("", task_views.TasksList.as_view(), name="all"),
    path("<int:pk>", task_views.TasksDetail.as_view(), name="single"),
    path("<username>", task_views.TasksUser.as_view(), name="user_tasks"),
    path("new/", task_views.TasksCreate.as_view(), name="create"),
    path("delete/<int:pk>", task_views.TasksDelete.as_view(), name="delete"),
]
