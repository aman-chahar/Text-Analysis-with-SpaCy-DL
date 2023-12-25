**Text Processing and NLP Exploration in Python**

I recently worked on a Python script that showcases a variety of Natural Language Processing (NLP) techniques using popular libraries. Let me walk you through the key tasks:

**Task 1: Import necessary libraries**
```python
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from spacy import displacy
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
nltk.download('wordnet')
```

In this task, we import essential libraries for data manipulation, tokenization, stemming, lemmatization, named entity recognition, Word2Vec modeling, and TF-IDF calculations.

**Task 2: Load the dataset**
```python
file_path = "/content/drive/MyDrive/Colab Notebooks/BBC_DATA.csv"
df = pd.read_csv(file_path)
display(df.head())
```

Here, we load a dataset from a CSV file into a Pandas DataFrame, assuming it contains news articles.

**Task 3: Tokenization with NLTK**
```python
sample_article = df.iloc[0, 1]
tokens_words = word_tokenize(sample_article)
tokens_sentences = sent_tokenize(sample_article)

print("\nTokenization with NLTK:")
print("Tokenized Words:", tokens_words)
print("Tokenized Sentences:", tokens_sentences)
```

Tokenization is the process of breaking down text into words and sentences. NLTK's tokenization functions are employed to achieve this.

**Task 4: Stemming and Lemmatization with NLTK**
```python
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

stemmed_words = [porter_stemmer.stem(word) for word in tokens_words]
lemmatized_words = [wordnet_lemmatizer.lemmatize(word) for word in tokens_words]

print("\nStemming with NLTK:", stemmed_words)
print("Lemmatization with NLTK:", lemmatized_words)
```

Stemming and lemmatization are applied to reduce words to their root forms using NLTK.

**Task 5: Named Entity Recognition with SpaCy**
```python
nlp = spacy.load("en_core_web_sm")
doc = nlp(sample_article)
displacy.render(doc, style="ent", jupyter=True)
```

SpaCy is used for named entity recognition, identifying entities like persons and organizations in the sample article.

**Task 6: Word2Vec with Gensim**
```python
sentences = [word_tokenize(article) for article in df['Text']]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

sample_word_vector = word2vec_model.wv['sample']
print("\nWord2Vec with gensim - Vector representation of 'sample':", sample_word_vector)
```

Word2Vec, a word embedding technique, is applied to represent words as vectors. Gensim is used for training.

**Task 7: TF-IDF with scikit-learn**
```python
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Text'])

article1_index = 0
article2_index = 1
cosine_sim = cosine_similarity(tfidf_matrix[article1_index], tfidf_matrix[article2_index])

print("\nTF-IDF with scikit-learn - Cosine Similarity between Article 1 and Article 2:", cosine_sim[0][0])
```

TF-IDF, a numerical statistic reflecting word importance, is calculated using scikit-learn. Cosine similarity is then computed between two news articles based on their TF-IDF representations.

This script provides a hands-on exploration of various NLP techniques, from basic text processing to advanced word embeddings and document similarity calculations. Feel free to use and modify it for your NLP projects! 🚀 #NLP #Python #DataScience #MachineLearning

---
