
# coding: utf-8

# In[612]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import os
import pickle

import pymysql
import json

config_fn = './config.json'



print("Import Complete")


# In[613]:


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


# In[614]:


def connect(config):
    return pymysql.connect(
        host=config['ai_db_host'],  # Database host
        port=config['ai_db_port'],  # Database port
        user=config['ai_db_username'],  # Database user
        passwd=config['ai_db_password'],  # Database password
        db=config['ai_db_name'],  # Database name
        connect_timeout=5,
        cursorclass=pymysql.cursors.DictCursor
    )

def pull_data():
    with open(config_fn, "r") as f:
        config = json.loads(f.read())
    conn = connect(config)
    sql_1 = "SELECT rowId, question, category FROM cleanHotlineQuestionAnswer;"
    with conn.cursor() as cursor:
        cursor.execute(sql_1)
    result = cursor.fetchall()
    cursor.close()
    return result


# In[615]:


def cluster(df, df2, N, name, v=False):
    clusterer = KMeans(n_clusters=N)
    clusterer.fit(df2)
    save_obj(clusterer, 'clusterer_' + name )

    transform = clusterer.transform(df2)
    
    d_center = []
    cluster = []
    for x in transform:
        d_center.append(min(x)**2)
        cluster.append(np.argmin(x))
    df['cluster'] = cluster
    df['d_from_center'] = d_center
    d_center = np.array(d_center)
    mean = np.mean(d_center)
    std = np.std(d_center)
    
    if v == True:
        print("Mean: {}".format(round(mean, 3)))
        print("STD: {}".format(round(std, 3)))
        print("")
        for cgroup in range(N):
            group = df.groupby('cluster').get_group(cgroup)
            print_clusters(group)

    return df

def print_clusters(group):
    std = np.std(list(group.d_from_center))
    print("Found {} messages of the same form.   STD: {}".format(len(group), std))
    for message in group.question.head(5):
        if group.question.count() > 1:
            print(message)
            print("")
    print("")


# In[616]:


def print_to_tsv(df, X, cat_name):
    vector_doc = 'doc_vectors_' + cat_name + '.tsv'
    count = 0
    with open(vector_doc,'w') as w:
        for question in X:
            string = ""
            for v in question:
                string = string + str(v) + "\t"
            w.write(string + os.linesep)
            count += 1
    w.close
    print("Wrote file {} with {} entries".format(vector_doc, count))


    meta_doc = 'doc_meta_' + cat_name + '.tsv'
    count = 0
    with open(meta_doc,'w') as w:
        w.write("cluster\tquestion\t" + os.linesep)
        for question, cluster in zip(list(df.question), list(df.cluster)):
            string = ""
            string = str(cluster) + "\t" + str(question) + "\t"
            w.write(string + os.linesep)  
            count += 1
    w.close
    print("Wrote file {} with {} entries".format(meta_doc, count))


# In[617]:


def train_model(df, N, name):
    print("Loaded {} Data Points".format(len(df)))
    vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.7 )
    X_vectoizer = vectorizer.fit_transform(list(df.question))
    save_obj(vectorizer, 'vectorizer_' + name )
    print("Vectorization Complete")

    n_components = 60
    explained_variance = 0.0
    while explained_variance < .5 and n_components < 175:
        svd = TruncatedSVD(n_components=n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(X_vectoizer)
        
        save_obj(svd, 'svd_' + name )
        save_obj(normalizer, 'normalizer_' + name )
        df["features"] = list(X)
        
        explained_variance = svd.explained_variance_ratio_.sum()
        n_components += 5
    print("Explained variance of the SVD step: {}%     n_componets: {}".format(
        int(explained_variance * 100), n_components))
    df = cluster(df, X, N, name, v=False)
    print_to_tsv(df, X, name)


# In[618]:


def train_all():    
    df_master = pd.DataFrame(pull_data())
    cat_names = ["Compensation", "Compliance", "Employee Benefits",
                "Leaves of Absence", "Recruiting and Hiring", "Terminations"]
    Ns = [20, 15, 14, 9, 9, 7]

    for name, N in zip(cat_names, Ns):
        df = df_master[df_master["category"] == name].copy()
        train_model(df, N, name)







# In[619]:


def predict(messages, category):
    vectorizer = load_obj( 'vectorizer_' + category )
    svd = load_obj( 'svd_' + category )
    normalizer = load_obj( 'normalizer_' + category )
    clusterer = load_obj("clusterer_" + category)

    
    pipeline =  make_pipeline(vectorizer, svd, normalizer)
    messages = pipeline.transform(messages)

    clusters = clusterer.predict(messages)
    return clusters


# In[620]:


message = ("With the minimum wage in MA going up to $11 per hour, for our sales people," +
"they make a base of $400 per week plus bonus." +
"Would we have to increase their base rate of pay to minimum wage? What if they work a week with no bonus?")
category = "Compensation"

print(predict([message], category))

