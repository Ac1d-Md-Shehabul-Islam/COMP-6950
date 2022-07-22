#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score

import json


# ### Read Data

# In[2]:


class readdata():
    
    def __init__(self,pathname):
        self.pathname = pathname
    
    def getDataFrame(self):
        df = pd.read_csv(self.pathname)
        return df
    
    def dropColumn(self, df):
        df.drop(['reviewerID','asin','reviewerName','helpful/0','helpful/1','summary','unixReviewTime','reviewTime'], axis=1, inplace=True)
        return df


# ### Data Preprocessing

# In[3]:


class preprocess():
    import string
    import nltk
#     nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')
    
    def __init__(self,dataframe):
        self.df = dataframe
        
    def rename(self):
        df.rename(columns={'reviewText':'feedback', 'overall':'rating'}, inplace=True)
        return df
    
    def addLabel(self):
        labels = []

        for i in df['rating']:
            if i<=2:
                labels.append('Negative')
            elif i==3:
                labels.append('Neutral')
            else:
                labels.append('Positive')
        output = pd.DataFrame(labels)
        df['label'] = output
        return df
    
    def getTextLen(self): # shows the length of words excluding whitespaces in a message body.
        df['text_length'] = df['feedback'].apply(len) 
        return df.head(4)
    
    def punc_count(self, txt):
        import string
        string.punctuation
        
        count = sum([1 for c in txt if c in string.punctuation])
        return (count/len(txt))*100
    
    def getPuncPerc(self):      
        df['punc_%'] = df['feedback'].apply(lambda x: self.punc_count(x))
        return df
    
    def remove_punc(self, txt):
        import string
        txt_nopunc = ''.join([i for i in txt if i not in string.punctuation])
        return txt_nopunc
    
    def getCleanText(self):
        df['clean_text'] = df['feedback'].apply(self.remove_punc) 
        return df
        
    def tokenize(self,txt):
        import re
        tokens = re.split('\W+',txt) # W means non-word characters and + means one or more
        return tokens
    
    def tokenizedCleanText(self):
        df['tokenized_clean_text'] = df['clean_text'].apply(lambda x: self.tokenize(x.lower()))
        return df

    def remove_stopwords(self,txt):
        import nltk
#         nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words('english')
#         print(stopwords[0:10])
        
        clean_msg = [word for word in txt if word not in stopwords]
        return clean_msg
    
    def textNoStopWord(self):
        df['text_no_stopword'] = df['tokenized_clean_text'].apply(self.remove_stopwords)
        return df
    
    def stemming(self, txt): # PorterSemmer is a popular one
        import nltk
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        
        text = [ps.stem(word) for word in txt]
        return text
    
    def ps_stem(self):
        df['ps_stem'] = df['text_no_stopword'].apply(self.stemming)
        df
        
    def lemmatization(self, txt):
        import nltk
        from nltk.stem import WordNetLemmatizer
        wn = nltk.WordNetLemmatizer()
#         nltk.download('wordnet')
        
        text = [wn.lemmatize(word) for word in txt]
        return text
    
    def wn_lemm(self):
        df['wn_lemmatize'] = df['text_no_stopword'].apply(self.lemmatization)
        return df
    
    def processText(self):
        import re
        import nltk
        from nltk.corpus import stopwords

        lemmatizer = nltk.WordNetLemmatizer()
        words = stopwords.words('english')
        df['processed_text'] = df['clean_text'].apply(lambda x: " ".join([lemmatizer.lemmatize(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
        df
        return df


# ### Data Visualization

# In[4]:


class view():

    def __init__(self,dataframe):
        self.df = dataframe
      
    def plot_rating(self):
        plt.hist(df['label'], color = 'lightblue', edgecolor = 'black')
        plt.title('Output')
        plt.xlabel('Reviews')
        plt.ylabel('Frequency')
        plt.show()
        
    def plot_label(self):
        plt.figure(figsize=(8, 8))
        plt.pie(df['label'].value_counts(), labels=["Positive", "Negative",'Neutral'], autopct='%.1f%%',colors = ['lightblue','orange'])
        plt.title("Output Label Distribution")
        plt.show() 
        
    def wordCompare(self): # We can clearly see that Positive have a high number of words as compared to Negatives. So it’s a good feature to distinguish.
        bins = np.linspace(0, 200, 40)

        plt.hist(df[df['label']=='Positive']['text_length'], bins, alpha=0.5, label='Positive')
        plt.hist(df[df['label']=='Negative']['text_length'], bins, alpha=0.5, label='Negative')
        plt.xlabel('text_length')
        plt.ylabel('frequency')
        plt.legend(loc='upper left')
        plt.show()
       
    def puncCompare(self): # Positive has a percentage of punctuations but not that far away from Negative. Surprising as at times Positive feedbacks can contain a lot of punctuation marks. But still, it can be identified as a good feature.
        bins = np.linspace(0, 50, 50)

        plt.hist(df[df['label']=='Positive']['punc_%'], bins, alpha=0.5, label='Positive')
        plt.hist(df[df['label']=='Negative']['punc_%'], bins, alpha=0.5, label='Negative')
        plt.xlabel('punc_percentage')
        plt.ylabel('frequency')
        plt.legend(loc='upper right')
        plt.show()


# ### Train-Test Split 

# In[5]:


class train_test():
    
    def __init__(self,dataframe):
        self.df = dataframe  
        self.X = self.df['processed_text']
        self.y = self.df['label']
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        
    def split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.df['processed_text'], self.y, test_size = 0.3, random_state=100)
        print(df.shape); print(X_train.shape); print(X_test.shape)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        return y_train, y_test

    def tfidf_vec(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
        train_tfIdf = vectorizer_tfidf.fit_transform(self.X_train.values.astype('U'))
        test_tfIdf = vectorizer_tfidf.transform(self.X_test.values.astype('U'))

#         print(vectorizer_tfidf.get_feature_names()[:10])
        return train_tfIdf, test_tfIdf


# ### Models

# In[6]:


class model():
    
    def __init__(self,dataframe):
        self.df = dataframe  
    
    def naive_bayes(self, a, b, c, d):
        import pandas as pd
        from sklearn.naive_bayes import MultinomialNB
        from sklearn import metrics

        nb_classifier = MultinomialNB()

#         nb_classifier.fit(train_tfIdf, y_train)
        nb_classifier.fit(a, c)

#         predNB = nb_classifier.predict(test_tfIdf) 
        predNB = nb_classifier.predict(b)
        print(predNB[:10])

#         Conf_metrics_tfidf = metrics.confusion_matrix(y_test, predNB, labels=['Positive', 'Neutral', 'Negative'])
        Conf_metrics_tfidf = metrics.confusion_matrix(d, predNB, labels=['Positive', 'Neutral', 'Negative'])
        print(Conf_metrics_tfidf)
        
        from sklearn.metrics import classification_report, confusion_matrix

#         print(confusion_matrix(y_test,predNB))
        print(confusion_matrix(d,predNB))
#         print(classification_report(d,predNB))
        a = classification_report(d,predNB, output_dict=True)
        df = pd.DataFrame(a)
        df.to_csv('naive_bayes.csv', index=False)
        
        
    def random_forest(self, a, b, c, d):
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn import metrics
        from sklearn.metrics import confusion_matrix

        rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 100)

#         rf_classifier.fit(train_tfIdf, y_train)
        rf_classifier.fit(a, c)
        
#         predRF = rf_classifier.predict(test_tfIdf) 
        predRF = rf_classifier.predict(b)
        print(predRF[:-2])

        # Calculate the accuracy score
#         accuracy_RF = metrics.accuracy_score(y_test, predRF)
        accuracy_RF = metrics.accuracy_score(d, predRF)
        print(f'Accuracy: {accuracy_RF*100} %')

#         Conf_metrics_RF = metrics.confusion_matrix(y_test, predRF, labels=['Positive', 'Neutral', 'Negative'])
        Conf_metrics_RF = metrics.confusion_matrix(d, predRF, labels=['Positive', 'Neutral', 'Negative'])
        print(Conf_metrics_RF)
        
        from sklearn.metrics import classification_report, confusion_matrix

#         print(confusion_matrix(y_test,predRF))
#         print(classification_report(y_test,predRF))
        print(confusion_matrix(d,predRF))
        a = classification_report(d,predRF, output_dict=True)
        df = pd.DataFrame(a)
        df.to_csv('random_forest.csv', index=False)

