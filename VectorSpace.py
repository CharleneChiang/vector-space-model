from pprint import pprint
from Parser import Parser
import util
import nltk
import numpy as np
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()


class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers.
    A document is represented as a vector. Each dimension of the vector corresponds to a
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    # Collection of document term vectors
    documentVectors = {}

    # Mapping of vector index to keyword
    vectorKeywordIndex = []

    # Tidies terms
    parser = None

    def __init__(self, documents={}, method="0"):
        self.documentVectors = {}
        self.documentList = list(documents.values())
        self.parser = Parser()
        self.documents = documents
        if(len(documents) > 0):
            self.build(documents, method)

    def build(self, documents, method):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(
            list(documents.values()))
        for key, value in documents.items():
            self.documentVectors[key] = self.makeVector(value, method)

        # print(self.vectorKeywordIndex)
        # print(self.documentVectors)

    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        # Mapped documents into a single word string
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        # Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex = {}
        offset = 0
        # Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word] = offset
            offset += 1
        return vectorIndex  # (keyword:position)

    def makeVector(self, wordString, method):
        """ @pre: unique(vectorIndex) """

        # Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        wordSet = set(wordList)
        if method == "0":
            for word in wordList:
                # Use simple Term Count Model
                vector[self.vectorKeywordIndex[word]] += 1/len(wordList)
        else:
            for word in wordSet:
                tfidf = util.tfidf(word, wordList, self.documentList)
                vector[self.vectorKeywordIndex[word]] += tfidf
        return vector

    def buildSubVector(self, wordStringID, method="1"):
        wordString = self.documents[wordStringID]
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        nltk_tags = nltk.pos_tag(wordList)
        wordResult = []
        for word in nltk_tags:
            if word[1] == "VB" or word[1] == "NN":
                wordResult.append(word[0])
        return np.array(self.makeVector(''.join(wordResult), method))*0.5

    def buildQueryVector(self, termList, method):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList), method)
        return query

    def related(self, documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        rating_dic = {}
        for key, value in self.documentVectors.items():
            ratings = util.cosine(
                self.documentVectors[documentId], value)
            rating_dic[key] = ratings
        return rating_dic

    def search(self, searchList, method="0"):
        """ search for documents that match based on a list of terms """
        rating_dic = {}
        queryVector = self.buildQueryVector(searchList, method)
        for key, value in self.documentVectors.items():
            rating_dic[key] = util.cosine(queryVector, value)

        result = {k: v for k, v in sorted(
            rating_dic.items(), key=lambda item: item[1], reverse=True)}

        return list(result.items())[:10]

    def search_eul(self, searchList, method="0"):
        """ search for documents that match based on a list of terms """
        rating_dic = {}
        queryVector = self.buildQueryVector(searchList, method)
        for key, value in self.documentVectors.items():
            rating_dic[key] = util.euclidean(queryVector, value)

        result = {k: v for k, v in sorted(
            rating_dic.items(), key=lambda item: item[1], reverse=True)}

        return list(result.items())[:10]

    def search_tfidf(self, searchList, method="1"):
        rating_dic = {}
        queryVector = self.buildQueryVector(searchList, method)
        for key, value in self.documentVectors.items():
            rating_dic[key] = util.cosine(queryVector, value)

        result = {k: v for k, v in sorted(
            rating_dic.items(), key=lambda item: item[1], reverse=True)}

        return list(result.items())[:10]

    def search_tfidf_eul(self, searchList, method="1"):
        rating_dic = {}
        queryVector = self.buildQueryVector(searchList, method)
        for key, value in self.documentVectors.items():
            rating_dic[key] = util.euclidean(queryVector, value)

        result = {k: v for k, v in sorted(
            rating_dic.items(), key=lambda item: item[1], reverse=True)}

        return list(result.items())[:10]

    def search_nltk(self, searchList, method="1"):
        rating_dic = {}
        for key, value in self.documentVectors.items():
            rating_dic[key] = util.cosine(searchList, value)

        result = {k: v for k, v in sorted(
            rating_dic.items(), key=lambda item: item[1], reverse=True)}

        return list(result.items())[:10]


if __name__ == '__main__':
    # test data
    documents = ["The cat in the hat disabled",
                 "A cat is a fine pet ponies.",
                 "Dogs and cats make good pets.",
                 "I haven't got a hat."]

    vectorSpace = VectorSpace(documents)

    # print(vectorSpace.vectorKeywordIndex)

    # print(vectorSpace.documentVectors)

    print(vectorSpace.related(1))

    # print(vectorSpace.search(["cat"]))

###################################################
