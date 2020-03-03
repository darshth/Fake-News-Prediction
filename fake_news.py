#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np 
import pandas as pd 
train=pd.read_csv('/Users/darshthakkar/Downloads/fake-news/train.csv')
test=pd.read_csv('/Users/darshthakkar/Downloads/fake-news/test.csv')
#0: Reliable
#1: Unreliable


# In[18]:


train.info()
test.info()


# In[19]:


train['title'] = train['title'].astype(str)
train['title'] = train['title'].str.lower() 
train["title"] = train['title'].str.replace('[^\w\s]','')


# In[20]:


train.head()


# In[21]:


test['title'] = test['title'].str.lower() 
test["title"] = test['title'].str.replace('[^\w\s]','')


# In[22]:


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


# In[23]:


test.head()


# In[24]:


#head for data labelled Unreliable
label1  = train.loc[train.label == 1]
label1.info()
label1.head(10)


# In[25]:


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


# In[26]:


#head for data labelled Unreliable
label0  = train.loc[train.label == 0]
label0.info()
label0.head(10)


# In[27]:


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


# In[28]:


#Checking Nulls
test.isnull().sum()
train.isnull().sum()


# In[29]:


#Filling Nulls
test=test.fillna(' ')
train=train.fillna(' ')


# In[30]:


#Concating all the columns to form a field- total, with title, author and text
test['total']=test['title']+' '+test['author']+test['text']
train['total']=train['title']+' '+train['author']+train['text']


# In[31]:


#Importing Tfidf and CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[32]:


#Importing Tfidf and CountVectorizer
transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1, 2))
counts = count_vectorizer.fit_transform(train['total'].values)
tfidf = transformer.fit_transform(counts)
targets = train['label'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf, targets, random_state=0)


# In[33]:


#Checking Extratrees model Accuracy
from sklearn.ensemble import ExtraTreesClassifier
                    
Ex = ExtraTreesClassifier(n_estimators=5,n_jobs=4)
Ex.fit(X_train, y_train)
print('Accuracy of Extratrees classifier: {:.2f}'
     .format(Ex.score(X_test, y_test)))


# In[34]:


#Checking RandomForest model Accuracy
from sklearn.ensemble import RandomForestClassifier

Rf = RandomForestClassifier(n_estimators=5,n_jobs=4)
Rf.fit(X_train, y_train)
print('Accuracy of RandomForest classifier: {:.2f}'
     .format(Rf.score(X_test, y_test)))


# In[35]:


#Checking AdaBoost model Accuracy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

Ad= AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=5)
Ad.fit(X_train, y_train)
print('Accuracy of Adaboost classifier: {:.2f}'
     .format(Ad.score(X_test, y_test)))


# In[36]:


#Checking Multinomial Naive Bayes model Accuracy
from sklearn.naive_bayes import MultinomialNB

NB = MultinomialNB()
NB.fit(X_train, y_train)
print('Accuracy of NB: {:.2f}'
     .format(NB.score(X_test, y_test)))


# In[37]:


#Preparing Test Data for Predictions
print(test.shape)
test_data = test.copy()
print(test_data.shape)
test_counts = count_vectorizer.transform(test['total'].values)
test_tfidf = transformer.fit_transform(test_counts)


# In[38]:


#Applying Adaboost model to predict the labels of test.csv
AdabM = Ad.predict(test_counts)
predictions = pd.DataFrame({'id':test.id, 'label':AdabM})
predictions.shape


# In[39]:


predictions.head()


# In[40]:


test.head()


# In[41]:


test = test.drop(['total'], axis = 1)


# In[42]:


test['label'] = predictions['label']


# In[43]:


test.head()


# In[44]:


test.to_csv('/Users/darshthakkar/Downloads/fake-news/testpredictions.csv')

