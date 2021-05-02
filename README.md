# Usage

`python3 main.py --query <query>`
e.g.
`python3 main.py --query "Trump"`

# Files

1. **Parser.py**

   clean and tokenize documents

2. **PorterStemmer.py**

   This is the Porter stemming algorithm,

   ported to Python from the version coded up in ANSI C by the author

3. **VectorSpace.py**

   - A algebraic model for representing text documents as vectors of identifiers

   a. **documentVectors** : Collection of all the document term vectors and their NewsID, based on the documents in **EnglishNews**

   b. **documentList** : Collection of all the document term vectors

   c.**parser** : given a function from Parser.py

4. **english.stop**

   english stop words collection

5. **util.py**

   includes utilities like removeDuplicates, cosine similarity, euclidean distance, n_containing, tf, idf weighting

   - **removeDuplicates(list)** : remove duplicates from a list

   - **cosine(vector1,vector2)** : related documents j and q are in the concept space by comparing the vectors

   cosine = ( V1 \* V2 ) / ||V1|| x ||V2||

   - **euclidean(vector1,vector2)** : related documents j and q are in the concept space by comparing the distance between vectors

   distance.euclidean(vector1, vector2)

   - **n_containing(word, documents)** : sum all documents which contains a specific word

   - **tf(word,wordList)** : simply count frequency of a word in a wordlist divided by the summation frequency of all the words in order to modify the value based on the document length

   - **idf(word, documents)** : log(len(documents) / (1 + n_containing(word, documents)))

   - **tfidf(word, wordList, documents)** : tf(word, wordlist) \* idf(word, documents)

6. **main.py**

   - iterate the file and read the docs

   - compute the final vector by combining the vector of original query and the vector of feedback query in different weights

   1 _ original query + 0.5 _ feedback query

   - main exacutable function
