from django.shortcuts import render
from nltk.corpus.reader import reviews
import numpy as np
import pickle
import pandas
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
import imdb
from bs4 import BeautifulSoup
import requests
from sqlalchemy import true
nltk.download('stopwords')
# Create your views here
stopset = set(stopwords.words('english'))
with open('models/ectorizer.pickle', 'rb') as efile:
    vectorizer = pickle.load(efile)

model = joblib.load('models/svc.pkl')

def index(request):
    context={'a':1, 'results':0}
    return render(request,'index.html',context)

def sentiment(movie):
    global reviews
    reviews= []
    sentiments={}
    ia = imdb.IMDb()
    search = ia.search_movie(movie)
    id = search[0].movieID
    page = requests.get('https://www.imdb.com/title/tt{}/reviews?ref_=tt_urv'.format(id))
    soup = BeautifulSoup(page.content, 'html.parser')
    movie_data=soup.find_all('div',attrs= {'class': 'lister-item-content'})
    for store in movie_data:
        review = store.find('a', class_ = 'title').text.replace('\n', '')
        reviews.append(review)
    for i in reviews:
        movie_vector=vectorizer.transform([i])
        pred = model.predict(movie_vector)
        if pred==0:
            pred=" 	\U0001f641"
        else:
            pred=" \U0001f600"
        sentiments[i]=pred
    return sentiments
with open('models/ew.pickle', 'rb') as handle:
    new = pickle.load(handle)
with open('models/similarity.pickle', 'rb') as efile:
    similarity = pickle.load(efile)

def recommend(movie):
    l=[]
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        #print(new.iloc[i[0]].title)
        l.append(new.iloc[i[0]].title)
    
    return l
        
def predict(request):
    movie=request.POST['movie']
    movies=recommend(movie)
    sentiments=sentiment(movie)
    return render(request,'index.html',{"movies":movies,"sentiments":sentiments,'results':1})
    # return render(request,'movie.html')
    


