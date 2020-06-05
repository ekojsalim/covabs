import pandas as pd
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer


class Answerer:
    def __init__(self):
        import nltk
        import ssl

        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download("stopwords")
        nltk.download('punkt')

        from nltk.corpus import stopwords

        self.abstracts = pd.read_csv("./data/abstracts.csv")
        self.stopwords = stopwords.words("english")
        self.porter = PorterStemmer()

    def build_model(self):
        # List of documents
        abstracts_texts = self.abstracts['Abstract'].values.tolist()
        docs = [[self.porter.stem(w.lower()) for w in word_tokenize(
            text) if w.isalpha() and w not in self.stopwords] for text in abstracts_texts]

        self.dictionary = gensim.corpora.Dictionary(docs)
        self.corpus = [self.dictionary.doc2bow(gen_doc) for gen_doc in docs]

        self.tf_idf = gensim.models.TfidfModel(self.corpus)
        self.sims = gensim.similarities.Similarity(
            "./working/", self.tf_idf[self.corpus], num_features=len(self.dictionary))

        self.built = True

    def answer(self, query):
        if self.built:
            tokens = sent_tokenize(query)
            query_doc_bow = []

            for line in tokens:
                query_doc = [self.porter.stem(w.lower()) for w in word_tokenize(
                    line) if w.isalpha() and w not in self.stopwords]
                query_doc_bow += self.dictionary.doc2bow(query_doc)

            #### Retrieve & Display Most Similar Abstracts ####
            query_doc_tf_idf = self.tf_idf[query_doc_bow]
            results = self.abstracts.copy()
            results["Similarity"] = self.sims[query_doc_tf_idf]
            results = results.sort_values(by="Similarity", ascending=False)

            return results
        else:
            raise Exception("Model not yet built")
