from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Query(models.Model):
    user=models.ForeignKey(User,on_delete=models.CASCADE,null=True,blank=True)
    question=models.TextField()
    response=models.TextField()
    timestamp=models.DateTimeField(auto_now_add=True)
    def __str__(self):
       return f"{self.user.username if self.user else 'Anonymous'} - {self.question[:50]}"
