from django.shortcuts import render
from django.views.generic.base import View
# Create your views here.
from picture_interface.search import get_img_src

from django.core.cache import cache


class IndexView(View):
    def get(self, request):
        model_data = cache.get('model_data')

        query_string = request.GET.get('s', "")
        img_list = []
        msg = ''
        if query_string:
            try:
                img_list, model_data = get_img_src(query_string, model_data)
                cache.set('model_data', model_data)
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