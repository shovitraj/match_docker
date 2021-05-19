
import os
import string
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from tika import parser
import string
import utils

ps = nltk.PorterStemmer()
sw = set(stopwords.words('english'))

def preprocess(text):
    punct = "".join(string.punctuation + str('\n●'))
    text = "".join([char.lower() for char in text if char not in punct])
    tokens = word_tokenize(text)
    # remove all tokens that are not alphabetic
    wordsisalpha = [word for word in tokens if word.isalpha()]
    
    word_count = Counter(wordsisalpha)
    return word_count

def cosSimilarity(x, y):
    dot_product = np.dot(x,y)
    normx = np.linalg.norm(x)
    normy = np.linalg.norm(y)
    similarity = dot_product / (normx * normy)
    return similarity

def Similarity(dict1, dict2):
    words_list = []
    for key in dict1:
        words_list.append(key)
    for key in dict2:
        words_list.append(key)
    list_size = len(words_list)
    
    v1 = np.zeros(list_size, dtype= np.int)
    v2 = np.zeros(list_size, dtype= np.int)
    
    i = 0
    for (key) in words_list:
        v1[i] = dict1.get(key,0)
        v2[i] = dict2.get(key,0)
        i = i+1
    return cosSimilarity(v1, v2)
