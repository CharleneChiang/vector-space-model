import sys
from scipy.spatial import distance
import math


# http://www.scipy.org/
try:
    from numpy import dot
    from numpy.linalg import norm
except:
    print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
    sys.exit()


def removeDuplicates(list):
    """ remove duplicates from a list """
    return set((item for item in list))


def cosine(vector1, vector2):
    """ related documents j and q are in the concept space by comparing the vectors :
            cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
    return float(dot(vector1, vector2) / (norm(vector1) * norm(vector2)))


def euclidean(vector1, vector2):
    return float(distance.euclidean(vector1, vector2))


def tf(word, wordlist):
    return wordlist.count(word)/len(wordlist)


def n_containing(word, documents):
    return sum(1 for doc in documents if word in doc)


def idf(word, documents):
    return math.log(len(documents) / (1 + n_containing(word, documents)))


def tfidf(word, wordlist, documents):
    return tf(word, wordlist) * idf(word, documents)
