import time

from django.shortcuts import render
from django.views.generic.base import View
# Create your views here.
from picture_interface.search import get_img


class IndexView(View):

    def get(self, request):

        query_string = request.GET.get('s', "")
        img_list = []
        msg = ''
        if query_string:
            try:
                img_list = get_img(query_string)
            except Exception as e:
                msg = 'Couldn\'t get images due to an error: ' + str(e)
        return render(request, "index.html", {
            "search": query_string,
            "img_list": img_list,
            "msg": msg,
        })


class InfoView(View):
    def get(self, request):
        return render(request, "info.html", {
        })

    def post(self,request):
        img = request.POST.get("img", "")
        return render(request, "info.html", {
            "img": img,
        })