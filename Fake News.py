#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np 
import pandas as pd 
train=pd.read_csv('/Users/darshthakkar/Downloads/fake-news/train.csv')
test=pd.read_csv('/Users/darshthakkar/Downloads/fake-news/test.csv')
#0: Reliable
#1: Unreliable


# In[12]:


train.info()
test.info()


# In[13]:


test['label']='t'


# In[14]:


train['title'] = train['title'].astype(str)
train['title'] = train['title'].str.lower() 
train["title"] = train['title'].str.replace('[^\w\s]','')


# In[15]:


train.head()


# In[16]:


test['title'] = test['title'].str.lower() 
test["title"] = test['title'].str.replace('[^\w\s]','')


# In[17]:


#Wordcloud for the train.csv dataset.
import matplotlib.pyplot as plt
from wordcloud import WordCloud
text = " ".join(train['title'])
wordcloud = WordCloud().generate(text)
plt.figure()
plt.subplots(figsize=(15,12))
wordcloud = WordCloud(
    background_color="white",
    max_words=len(text),
    max_font_size=40,
    relative_scaling=.5).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[18]:


test.head()


# In[19]:


#head for data labelled Unreliable
label1  = train.loc[train.label == 1]
label1.info()
label1.head(10)


# In[20]:


#Word Cloud for News labelled Unreliable
import matplotlib.pyplot as plt
from wordcloud import WordCloud
text = " ".join(label1['title'])
wordcloud = WordCloud().generate(text)
plt.figure()
plt.subplots(figsize=(15,12))
wordcloud = WordCloud(
    background_color="white",
    max_words=len(text),
    max_font_size=40,
    relative_scaling=.5).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[21]:


#head for data labelled Unreliable
label0  = train.loc[train.label == 0]
label0.info()
label0.head(10)


# In[22]:


#Wordcloud for data labelled Reliable
text = " ".join(label0['title'])
wordcloud = WordCloud().generate(text)
plt.figure()
plt.subplots(figsize=(15,12))
wordcloud = WordCloud(
    background_color="white",
    max_words=len(text),
    max_font_size=40,
    relative_scaling=.5).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[23]:


#Checking Nulls
test.isnull().sum()
train.isnull().sum()


# In[24]:


#Filling Nulls
test=test.fillna(' ')
train=train.fillna(' ')


# In[25]:


#Concating all the columns to form a field- total, with title, author and text
test['total']=test['title']+' '+test['author']+test['text']
train['total']=train['title']+' '+train['author']+train['text']


# In[26]:


#Importing Tfidf and CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[27]:


#Importing Tfidf and CountVectorizer
transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1, 2))
counts = count_vectorizer.fit_transform(train['total'].values)
tfidf = transformer.fit_transform(counts)
targets = train['label'].values
test_counts = count_vectorizer.transform(test['total'].values)
test_tfidf = transformer.fit_transform(test_counts)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf, targets, random_state=0)


# In[28]:


#Applying Extratrees model
from sklearn.ensemble import ExtraTreesClassifier
                    
Ex = ExtraTreesClassifier(n_estimators=5,n_jobs=4)
Ex.fit(X_train, y_train)
print('Accuracy of Extratrees classifier: {:.2f}'
     .format(Ex.score(X_test, y_test)))


# In[29]:


#Applying RandomForest model
from sklearn.ensemble import RandomForestClassifier

Rf = RandomForestClassifier(n_estimators=5,n_jobs=4)
Rf.fit(X_train, y_train)
print('Accuracy of RandomForest classifier: {:.2f}'
     .format(Rf.score(X_test, y_test)))


# In[30]:


#Applying AdaBoost model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

Ad= AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=5)
Ad.fit(X_train, y_train)
print('Accuracy of Adaboost classifier: {:.2f}'
     .format(Ad.score(X_test, y_test)))


# In[31]:


#Applying Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB

NB = MultinomialNB()
NB.fit(X_train, y_train)
print('Accuracy of NB: {:.2f}'
     .format(NB.score(X_test, y_test)))

