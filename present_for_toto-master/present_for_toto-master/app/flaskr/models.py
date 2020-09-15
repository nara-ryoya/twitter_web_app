import numpy as np
import joblib
import MeCab
import pandas as pd
import re
import oseti
import glob
from sklearn.feature_extraction.text import CountVectorizer as CV 
from sklearn.feature_extraction.text import TfidfTransformer
import os
import urllib.request
from sklearn.linear_model import LogisticRegression


class SentimentAnalysis(object):
    def remove_symbol(text):
        removed = re.sub(r"[\W]+|[a-z]+|[0-9]+|[A-Z]+|ｗ+|_+|ç+|ë+|①+|②+|③+|④+|ō", " ", text)
        return removed

    def japanese_analyzer(string):
        tagger = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
        result_list = []
        for line in tagger.parse(string).split("\n"):
            splited_line = line.split("\t")
            if len(splited_line) >= 2 and "名詞" in splited_line[1]:
                result_list.append(splited_line[0])
        return result_list
    
    def download_stopwords(path):
        url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)
    
    def create_stopwords(file_path):
        stop_words = []
        for w in open(path, "r"):
            w = w.replace('\n','')
            if len(w) > 0:
              stop_words.append(w)
        return stop_words    

    def getName(label):
        print(label)
        if label == 0:
            return "いい"
        elif label == 1: 
            return "悪い"
        elif label == 2: 
            return "普通"
        else: 
            return "Error"
        
    def predict(tweet):
        # モデル読み込み
        model = joblib.load('./nn.pkl')
        removed_tweet = remove_symbol(tweet)
        path = "stop_words.txt"
        download_stopwords(path)
        stop_words = create_stopwords(path)
        cv = CV(stop_words=stop_words, analyzer=japanese_analyzer)
        feature_vectors = cv.fit_transform(removed_tweet)
        tfidf_transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
        tfidf = tfidf_transformer.fit_transform(feature_vectors.astype('f'))
        matrix = tfidf.toarray()
        pred = model.predict(matrix)
        hyouka = getName(pred)
        return hyouka




        
       


