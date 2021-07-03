import joblib
import streamlit as st
import xlrd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import texthero as hero
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity

xlrd.xlsx.ensure_elementtree_imported(False, None)
xlrd.xlsx.Element_has_iter = True

def score(y, X, titles, abstracts, images):
    recommended_images = []
    recommended_titles = []
    for i in range(len(titles)):
        similarity = cosine_similarity(y[0].reshape(1,-1), X[i].reshape(1,-1))
        if similarity>0.2:
            st.image(images[i])
            st.write("Paper Title: " + titles[i])
    return 0


def collectdata(sheet):
    total = sheet.nrows
    titles = []
    abstracts = []
    images = []
    for i in range(total):
        title = sheet.cell_value(i,0)
        titles.append(title)
        abstract = sheet.cell_value(i,1)
        abstracts.append(abstract)
        image =  sheet.cell_value(i,2)
        images.append(image)
    return titles, abstracts, images

st.markdown('# Researcher Spot')
#st.title("Researcher Spot")
st.header("A database of graphical abstracts of research papers")
#st.text("*includes only chemistry and material science papers currently")
st.subheader("Get ideas for your next research paper's TOC graphic")
st.text("              ")

query = st.text_input(label="Search with keywords")
if st.button(label="Get recommendations"):
    if not query:
        st.markdown("Please enter a query")
    else:
        tfidf_file = "articlesdb.xlsx"
        tfidf_wb = xlrd.open_workbook(tfidf_file)
        tfidf_sheet = tfidf_wb.sheet_by_index(0)
        tfidfmodel = joblib.load("tfidfmodel.pkl")
        tfidf_X = joblib.load("vectorspace.pkl")
        tfidf_titles, tfidf_abstracts, tfidf_images = collectdata(tfidf_sheet)
        tfidf_y = tfidfmodel.transform([query]).toarray()
        score(tfidf_y, tfidf_X, tfidf_titles, tfidf_abstracts, tfidf_images)

