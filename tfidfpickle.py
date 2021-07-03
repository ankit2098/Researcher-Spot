import xlrd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import texthero as hero
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
import joblib

xlrd.xlsx.ensure_elementtree_imported(False, None)
xlrd.xlsx.Element_has_iter = True

def textprocessing(text):
    text = re.sub('[^A-Za-z]+', ' ', text)
    lemmatizer = WordNetLemmatizer()
    text = lemmatizer.lemmatize(text)
    text = pd.Series(text)
    text = hero.clean(text)
    cleaned_text = hero.remove_stop_words(text)
    return cleaned_text[0]

def makecorpus(data):
    corpus = []
    for i in range(0, len(data)):
        corpus.append(textprocessing(data[i]))
    return corpus

def collectdata(sheet):
    total = sheet.nrows
    data = []
    titles = []
    abstracts = []
    for i in range(total):
        title = sheet.cell_value(i,0)
        titles.append(title)
        abstract = sheet.cell_value(i,1)
        abstracts.append(abstract)
        summary = title + "." + abstract # concatinating title and abstract
        data.append(summary)
    return data

def tfidf(corpus):
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(corpus).toarray()
    print(X.shape)
    joblib.dump(tfidf, "tfidfmodel.pkl")
    joblib.dump(X, "vectorspace.pkl")
    return 0

tfidf_file = "articlesdb.xlsx"
tfidf_wb = xlrd.open_workbook(tfidf_file)
tfidf_sheet = tfidf_wb.sheet_by_index(0)
tfidf_data = collectdata(tfidf_sheet)
tfidf_corpus = makecorpus(tfidf_data)
tfidf(tfidf_corpus)
print("finished")