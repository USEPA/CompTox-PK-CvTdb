# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:52:13 2017

Based on https://github.com/japerk/nltk-trainer/blob/master/nltk_trainer/classification/scoring.py

@author: rsayre01
"""

import sys

# numerical analysis
import random
import numpy as np
import math
from statistics import mode, mean

# data organization and display
import collections
import pandas as pd
import matplotlib.pyplot as plt

# database access
import pymysql.cursors
from Bio import Entrez, Medline
import re

# machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.metrics.scores import precision, recall
from sklearn.neural_network import MLPClassifier

print('Fetching active PMIDs')     

## ! user input required
host_name = 'your_hostname'
username = 'your_username'
password = 'your_password'
db = 'CvTdb'

actives_pmids = []

connection = pymysql.connect(host=host_name,
                             user=username,
                             password=password,
                             db=db,
                             cursorclass=pymysql.cursors.DictCursor)

try:
    with connection.cursor() as cursor:
        sql = 'SELECT DISTINCT pmid from documents' #JOIN studies ON studies.fk_extraction_document_id = documents.id'
        cursor.execute(sql)
        result = cursor.fetchall()
        for record in result:
            actives_pmids.append(record['pmid'])
            
finally:
    connection.close()

print('Fetching abstracts')    
Entrez.email = 'your_email'

pmid_str = list(map(str, actives_pmids))

handle = Entrez.efetch(db="pubmed", id=pmid_str, rettype="medline", retmode="text")
records = Medline.parse(handle)
abstract_dict = {}
for record in records:     
    for x in record.keys():         
        if x == "AB":             
            abstract_dict[record['PMID']] = record['AB']       
handle.close()

actives_df = pd.DataFrame(list(abstract_dict.items()), columns=['pmid', 'abstract'])

print("Generating sample PMIDs")
pmid_ints = list(map(int,actives_df['pmid'].tolist()))
sample = random.sample(range(min(pmid_ints),max(pmid_ints)), 2*len(pmid_ints))
sample = list(map(str, sample))

print("Fetching abstracts for random sample")
handle = Entrez.efetch(db="pubmed", id=sample, rettype="medline", retmode="text")
records = Medline.parse(handle)

abstract_dict = {}
for record in records:     
    for x in record.keys():         
        if x == "AB":             
            abstract_dict[record['PMID']] = record['AB']      
handle.close()

results = pd.DataFrame(list(abstract_dict.items()), columns=['pmid', 'abstract'])
sample_df = results.sample(n=len(pmid_ints))
sample_text_lst = sample_df.abstract.tolist()

print('Vectorizing abstract training data')    
useful_abstracts = list(actives_df['abstract'])
random_abstracts = list(sample_df['abstract'])
useful_tuples = [("hit", x) for x in useful_abstracts]
random_tuples = [("swing", x) for x in random_abstracts]
abstract_set = useful_abstracts + random_abstracts
classified_set = useful_tuples + random_tuples
random.shuffle(classified_set)
                   
vect = TfidfVectorizer(min_df=1)
tfidf_useful = vect.fit_transform(useful_abstracts)
useful_mean = np.mean((tfidf_useful * tfidf_useful.T).A)

tfidf_not_useful = vect.fit_transform(random_abstracts)
not_useful_mean = np.mean((tfidf_not_useful * tfidf_not_useful.T).A)

def find_features(document):
    words = re.findall(r'\w+', document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

print('Finding informative features in curated abstracts')
all_words_in_set = []
for i in range(len(abstract_set)):
    words = re.findall(r'\w+', abstract_set[i])
    all_words_in_set.append(words)
all_words_in_set_lst = [val for sublist in all_words_in_set for val in sublist if len(val) > 3]
word_features = [x for x,y in collections.Counter(all_words_in_set_lst).most_common(int(math.sqrt(len(all_words_in_set_lst))))]
featuresets = [(find_features(abstr), category) for (category, abstr) in classified_set]

print('Training classifiers with 10-fold cross-validation')
MNB_classifier = SklearnClassifier(MultinomialNB())
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
SGDClassifier_classifier = SklearnClassifier(SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, max_iter=10, tol=None))
SVC_classifier = SklearnClassifier(SVC())
LinearSVC_classifier = SklearnClassifier(LinearSVC())
knn_classifier = SklearnClassifier(KNeighborsClassifier(3))
DTree_classifier = SklearnClassifier(DecisionTreeClassifier(max_depth=5))
RF = SklearnClassifier(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))
MLP = SklearnClassifier(MLPClassifier(alpha=1))
AdaBoost = SklearnClassifier(AdaBoostClassifier())
NuSVC_classifier = SklearnClassifier(NuSVC())

classifiers = [MNB_classifier,
               BernoulliNB_classifier,
               LogisticRegression_classifier,
               SGDClassifier_classifier,
               SVC_classifier,
               LinearSVC_classifier,
               NuSVC_classifier,
               knn_classifier,
               DTree_classifier,
               RF,
               MLP,
               AdaBoost]
               
num_folds = 10

active_subset = []
inactive_subset = []
for i in range(len(featuresets)):
    if featuresets[i][1] == 'hit':
        active_subset.append(featuresets[i])
    else:
        inactive_subset.append(featuresets[i])

active_subset_size = int(len(active_subset)/num_folds)
inactive_subset_size = int(len(inactive_subset)/num_folds)
subset_size = int(len(featuresets)/num_folds)

accuracies = []
confusion_matrices = []

for c in classifiers:
    accuracy_lst = []
    precision_act_lst = []
    recall_act_lst = []
    precision_inact_lst = []
    recall_inact_lst = []
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0  

    c_name = re.search(r'\((.*)\(', str(c)).group(1)
    for i in range(num_folds):
        training_this_round = inactive_subset[:i*inactive_subset_size] + inactive_subset[(i+1)*inactive_subset_size:] + active_subset[:i*active_subset_size] + active_subset[(i+1)*active_subset_size:]
        testing_this_round = inactive_subset[i*inactive_subset_size:][:inactive_subset_size] + active_subset[i*active_subset_size:][:active_subset_size]
        random.shuffle(training_this_round)
        random.shuffle(testing_this_round)        
        c.train(training_this_round)
        accuracy_lst.append(nltk.classify.accuracy(c, testing_this_round))
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        for i, (feats,label) in enumerate(testing_this_round):
            refsets[label].add(i)
            observed = c.classify(feats)
            testsets[observed].add(i)
        ref = dict(list(refsets.items()))
        test = dict(list(testsets.items()))
        precision_act_lst.append(precision(refsets['hit'], testsets['hit']))
        recall_act_lst.append(recall(refsets['hit'], testsets['hit']))
        precision_inact_lst.append(precision(refsets['swing'], testsets['swing']))
        recall_inact_lst.append(recall(refsets['swing'], testsets['swing']))
        for i in range(len(testing_this_round)):
            if i in refsets['hit'] and i in testsets['hit']:
                true_positives += 1
            elif i in refsets['hit'] and i in testsets['swing']:
                false_negatives += 1
            elif i in refsets['swing'] and i in testsets['hit']:
                false_positives += 1
            else:
                true_negatives += 1
    precision_act_result = [0 if x is None else x for x in precision_act_lst]
    precision_inact_result = [0 if x is None else x for x in precision_act_lst]
    data = {'active':(true_positives,false_negatives), 'inactive':(false_positives,true_negatives)}              
    accuracies.append((c_name,
                       "10-fold Cross-validated Accuracy: " + str(round(mean(accuracy_lst),3)),
                       "Precision (active): " + str(round(mean(precision_act_result),3)),
                       "Recall (active): " + str(round(mean(recall_act_lst),3)),
                       "Precision (inactive): " + str(round(mean(precision_inact_result),3)),
                       "Recall (inactive): " + str(round(mean(recall_inact_lst),3))
                       ))
    confusion_matrices.append((c_name,pd.DataFrame(data, index=['active','inactive'])))
    
cmap=plt.cm.RdYlGn              

print("Plotting classifier accuracies")
classifiers_gt90 = []
n_abstracts = len(classified_set)
len_confusion = len(confusion_matrices)
for i in range(len_confusion):
    wrong = confusion_matrices[i][1]['active']['inactive'] + confusion_matrices[i][1]['inactive']['active']
    if wrong/n_abstracts < 0.1:
        classifiers_gt90.append(classifiers[i])
    fig = plt.figure()   
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    df_confusion = confusion_matrices[i][1]        
    ax.text(0,0,df_confusion['active']['active'],fontweight='bold',ha='center', va='center', fontsize=15) #color="green",
    ax.text(0,1,df_confusion['active']['inactive'],fontweight='bold',ha='center', va='center', fontsize=15) #color="red",
    ax.text(1,0,df_confusion['inactive']['active'],fontweight='bold',ha='center', va='center', fontsize=15) #color="green",
    ax.text(1,1,df_confusion['inactive']['inactive'],fontweight='bold',ha='center', va='center', fontsize=15) #color="red",
    ax.matshow(df_confusion, cmap=cmap)
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns)
    plt.yticks(tick_marks, df_confusion.index)
    ax.set_ylabel("Predicted")
    ax.set_xlabel("Observed")
    ax.set_title(confusion_matrices[i][0])
    plt.show()

sys.exit()

classifiers = classifiers_gt90

print("Classifying all PubMed abstracts")
pmid_chunks = np.arange(0,31300000,100000)
positive_abstracts = []

for i in range(len(pmid_chunks)):
    print('Classifying chunk %s of %s' % str(i), str(len(pmid_chunks)))
    abstract_dict = {}
    val_sample = list(map(str, range(pmid_chunks[i],pmid_chunks[i+1])))
    for item in actives_pmids:
        while item in val_sample: val_sample.remove(item)
    print("Fetching abstracts for PMIDs")    
    handle = Entrez.efetch(db="pubmed", id=val_sample, rettype="medline", retmode="text")
    records = Medline.parse(handle)

    print("Fetching test abstracts")
    for record in records:     
        for x in record.keys():         
            if x == "AB":             
                abstract_dict[record['PMID']] = record['AB']      
    handle.close()

    val_results = pd.DataFrame(list(abstract_dict.items()), columns=['pmid', 'abstract'])
    
    classifications = []
    for j in range(len(val_results)):    
        votes = []        
        for c in gt90_classifiers:            
            features = find_features(val_results['abstract'][j])            
            vote = c.classify(features)            
            votes.append(vote)            
        try:                
            consensus = mode(votes)              
            choice_votes = votes.count(consensus)                
            conf = choice_votes / len(votes)                
            if consensus == "hit":
                if conf > 0.75:
                    positive_abstracts.append((val_results['pmid'][j],conf,val_results['abstract'][j]))            
        except:                
            continue                

abstract_classified_df = pd.DataFrame(classifications, columns=['pmid','abstract_class','confidence'])
abstract_classified_df.groupby(['abstract_class']).size()  

# inner join 

pos_abstract_pmid_lst = [x[0] for x in positive_abstracts]   

name_lst = []
    # get synonyms
connection = pymysql.connect(host='mysql-res1.epa.gov',
                             user='_dataminer',
                             password=dataminer_password,
                             db='ro_stg_dsstox',
                             cursorclass=pymysql.cursors.DictCursor)
try:
    with connection.cursor() as cursor:
        sql = 'SELECT preferred_name from generic_substances;' 
        cursor.execute(sql)
        result = cursor.fetchall()
        for record in result:
            name_lst.append(record['preferred_name'])

finally:
    connection.close()

name_lst = [n for n in name_lst if n.lower() not in ('water','based', 'serum',
            'potassium', 'sodium', 'escherichia coli', 'lipids', 'helium',
            'oxygen', 'carbon', 'fatty acids', 'glutamine','urea', 'calcium',
            'nitrogen', 'amino acids','oxide','cytochrome c', 'superoxide',
            'saline', 'hydrogen', 'magnesium', 'potassium', 'polysaccharide',
            'creatinine', 'lactic acid', 'pseudomonas putida','egg white',
            'collagen','cytokines','ovalbumin','sucrose','peanut oil','honey',
            'pseudomonas','olive oil',"soap",'basal medium','','graphite',
            'tumor necrosis factor-alpha','corn oil','vegetable oils',
            'gelatin','sawdust','labeled glucose','hydrogen cyanide (hcn)',
            "dulbecco's modified eagle's medium",'centa',"adenosine triphosphate",
            'adenosine','brass', 'adrenocorticotropic hormone', 'amylase',
            'cytochrome c', 'superoxide','albumin','ige','css','agar','agarose',
            'bovine serum albumin','bran','dex','catalase','elastase',
            'air','sand', 'adenosine triphosphatase','bovine growth hormone',
            'basic fibroblast growth factor',"ringer's lactate solution", 
            'insulin-like growth factor i','insulin','cholesterol',
            'bovine insulin','human growth hormone','human insulin','human serum albumin',
            'horse serum','hormone replacement therapy','human serum albumin')]

tent_name_match = []
#start = time.time()
for i in range(len(positive_abstracts)):
    this_abst = positive_abstracts['abstract'].iloc[i].lower()
    for j in range(len(name_lst)):
        this_name = name_lst[j].lower()
        if this_abst.find(this_name) == -1:
            continue
        else:
            tent_name_match.append((positive_abstracts['pmid'].iloc[i], this_name, j))
#end = time.time()
#elapsed = end - start
    
name_df = []
uniq = list(set([x[0] for x in tent_name_match]))
for u in uniq:
    this_lst = []
    for tup in tent_name_match:    
        if tup[0] == u:
            this_lst.append(tup[1][1:-1])
    name_df.append((u,this_lst))

chem_df = pd.DataFrame(name_df, columns=['pmid','name_lst'])    
res = pd.merge(positive_abstracts, chem_df, how='left',on='pmid')
res.to_csv('M:\\CvT\\tent_name_final.csv')  

u_name = tent_name_match.name.unique()
#u_name_df = pd.read_csv('M:\\CvT\\u_name.csv')
#u_name = list(u_name_df[u_name_df.columns[-1]].values)

name_cas_lst = []
    # get synonyms
    
for u in u_name:
    this_name = []
    connection = pymysql.connect(host='mysql-res1.epa.gov',
                             user='_dataminer',
                             password=dataminer_password,
                             db='ro_stg_dsstox',
                             cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            sql = 'SELECT preferred_name, casrn, dsstox_substance_id from generic_substances;' 
            cursor.execute(sql)
            result = cursor.fetchall()
            for record in result:
                name_cas_lst.append((record['preferred_name'], record['casrn'], record['dsstox_substance_id'])) 
    finally:
        connection.close()