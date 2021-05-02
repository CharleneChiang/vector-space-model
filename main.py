import os
import argparse
import numpy as np
import nltk
from VectorSpace import VectorSpace


files_path = os.listdir(os.getcwd()+"/EnglishNews")


def return_query():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="input query")

    return parser.parse_args()


def form(result):
    print('NewsID           Score')
    print('------------------------')
    # print(len(result))
    for i in range(len(result)):
        print(result[i])


if __name__ == '__main__':
    query = return_query().query.split(" ")
    print(query)
    document = {}

    for txt in files_path:
        directory_path = os.getcwd()+"/EnglishNews"
        txt_path = directory_path + '/' + txt
        with open(txt_path, "r") as f:
            document[txt] = f.read()

    vectorspace = VectorSpace(document)
    form(vectorspace.search(query))
    form(vectorspace.search_eul(query))
    form(vectorspace.search_tfidf(query))
    form(vectorspace.search_tfidf_eul(query))

    subqueryID = vectorspace.search_tfidf(query)[0][0]
    subqueryVector = vectorspace.buildSubVector(subqueryID)
    finalVector = np.array(vectorspace.buildQueryVector(
        query, method="1")) + subqueryVector
    form(vectorspace.search_nltk(finalVector))
