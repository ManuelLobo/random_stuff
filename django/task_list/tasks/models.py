from django.db import models
from django.urls import reverse
import misaka

from django.contrib.auth import get_user_model
User = get_user_model()


class Task(models.Model):
    user = models.ForeignKey(User, related_name='tasks',
                             on_delete=models.CASCADE)
    created_date = models.DateField(auto_now=True)
    title = models.CharField(max_length=100)
    message = models.TextField()
    # message_html = models.TextField(editable=False)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        # self.message_html = misaka.html(self.message)
        super().save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse('tasks:single', kwargs={#'username': self.user.username,
                                               'pk': self.pk})
