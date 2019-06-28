from django.shortcuts import render
from django.views.generic.base import View
# Create your views here.
from picture_interface.search import get_img_src

class IndexView(View):
    def get(self, request):
        s = request.GET.get('s', "")
        img_list=[]
        msg=''
        if s:
            try:
                img_list=get_img_src(s)
            except Exception as e:
                msg=str(e)+'NOT EXISTING'
        return render(request, "index.html", {
            "search":s,
            "img_list": img_list,
            "msg":msg,
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