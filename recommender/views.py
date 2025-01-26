import pandas as pd
from pandas.io.formats import style
from django.shortcuts import render
from recommender.recommendation_services import data_processing
from IPython.display import display
# Create your views here.


def home_page(request):
    context = {}
    if request.method == "POST":
        input_name = request.POST["r_name"]
        html_data = data_processing.recommend_restaurant(input_name)

        context = {
            'top_restaurant_data': html_data.to_html(),
        }

    return render(request, "main_page.html", context)
