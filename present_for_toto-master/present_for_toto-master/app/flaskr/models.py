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
from sklearn.model_selection import train_test_split



class SentimentAnalysis(object):
    
    @classmethod
    def remove_symbol(self, text):
        removed = re.sub(r"[\W]+|[a-z]+|[0-9]+|[A-Z]+|ｗ+|_+|ç+|ë+|①+|②+|③+|④+|ō", " ", text)
        return removed

    @classmethod
    def japanese_analyzer(self, string):
        tagger = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
        result_list = []
        for line in tagger.parse(string).split("\n"):
            splited_line = line.split("\t")
            if len(splited_line) >= 2 and "名詞" in splited_line[1]:
                result_list.append(splited_line[0])
        return result_list
    

    @classmethod
    def getName(self, label):
        print(label)
        if label == 0:
            return "いい"
        elif label == 1: 
            return "悪い"
        elif label == 2: 
            return "普通"
        else: 
            return "Error"
        
    @classmethod
    def predict(self, tweet):
        # モデル読み込み
        txt_dt=list(map(lambda x: x.strip("\n"), open("tweet.txt", "r").readlines()))
        removed_tweet = self.remove_symbol(tweet)
        txt_dt+=[removed_tweet]
        txt_df=pd.Series(txt_dt)
        negaposi_dt=list(map(lambda x: int(x.strip("\n")), open("negaposi.txt", "r").readlines()))
        negaposi_df=pd.Series(negaposi_dt)
        f = open('stop_words.txt')
        stop_words = list(map(lambda x: x.strip("\n"), f.readlines()))
        cv = CV(stop_words=stop_words, analyzer=self.japanese_analyzer)
        feature_vectors = cv.fit_transform(txt_df)
        tfidf_transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
        tfidf = tfidf_transformer.fit_transform(feature_vectors.astype('f'))
        matrix = tfidf.toarray()
        X_matrix = matrix[:-1]
        t_matrix = matrix[-1]
        X_train, X_test, y_train, y_test = train_test_split(X_matrix, negaposi_df, test_size=0.3, random_state=0)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        pred = lr.predict(t_matrix.reshape(1, -1))
        hyouka = self.getName(pred)
        return hyouka




        
       


