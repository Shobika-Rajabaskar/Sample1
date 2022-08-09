#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[23]:


amz=pd.read_csv("status_text.csv")
amz.sample(5)


# In[51]:


corpus = []
stop_words = set(stopwords.words("english"))
for i in range(1,1000):
    status_text = re.sub('[^a-zA-Z]', ' ', amz['status_text'][i])
    status_text = status_text.lower()
    status_text = status_text.split()
    ps = PorterStemmer()
    status_text = [ps.stem(word) for word in status_text
                if not word in set(stopwords.words('english'))]
    status_text = ' '.join(status_text) 
    corpus.append(status_text)
amz.head()


# In[45]:


corpus[0:10]


# In[46]:


vectorizer = CountVectorizer(stop_words='english')
#converting toarrray() to get a dense matrix
V_s = vectorizer.fit_transform(corpus).toarray()


# In[47]:


def idf_transform(word_col):
    #words present in how many documents - df
    w = len(word_col[np.nonzero(word_col)])
    #compute idf for a word
    return np.log(len(word_col)/(w + 1))

def tf_idf(bow):
    #TF matrix
    tf = np.log(bow + 1)
    #1d array
    idf = np.apply_along_axis(idf_transform,0,bow)
    #tf-idf
    return (np.multiply(tf,idf))


# In[48]:


V_s_tfidf = tf_idf(V_s)


# In[49]:


def kmeans(V,k):
    #select k reviews as centers
    k_center_i = random.sample(range(0,V.shape[0]),k)
    center_v = V[k_center_i, :]
    
    #all reviews
    A_i = np.array([x for x in range(0,V.shape[0])])
    all_v_norm = np.apply_along_axis(np.linalg.norm,1,V)
    
    #clusters - initial
    clusters = [None] * k
    clusters[0] = A_i.tolist()
    for i in range(1,k):
        clusters[i] = []
    j=0
    while True:
        
        print(j)
        for i in range(0,len(clusters)):
            print('cluster',i,len(clusters[i]))
#         #only printing the sizes of first 4 clusters
#         print("iteration",j,"cluster0",len(clusters[0]),"cluster1",len(clusters[1]),"cluster2",len(clusters[2]),
#               "clusters3",len(clusters[3]))
        #Norm of cluster center vectors
        center_v_norm = np.apply_along_axis(np.linalg.norm,1,center_v)
        #Cosine similarity: 
        #x @ y
        product_v = V @ np.transpose(center_v)
        #divide by norms ||x|| and ||y||
        product_v_n = np.apply_along_axis(np.true_divide,1,product_v,center_v_norm)
        product_v_norm = np.apply_along_axis(np.true_divide,0,product_v_n,all_v_norm)
        #get each review has maximum cosine similarity with which center
        max_center = np.argmax(product_v,axis=1)

        #assign to closest clusters
        clusters_new = [None] * k
        for i in range(k):
            r = np.where(np.array(max_center) == i)
            clusters_new[i] = r[0].tolist()

        if (np.array_equal(clusters,clusters_new)):
            break
        else:
            j = j+1
        
        #calculate new centers
        for i in range(k):
            reviews = V[clusters_new[i], :]
            center_v[i] = np.mean(reviews,axis=0)
        
        #set old clusters as new clusters
        clusters = clusters_new.copy()
    
    print("Clusters converged after",j+1,"iterations")
    return clusters


# In[41]:


clusters = kmeans(V_s_tfidf,20)


# In[53]:


for i in range(0,len(clusters)):
    print(amz.iloc[clusters[i]][['status_text']].sample(7))

