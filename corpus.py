from typing import List, Dict
import math

class Document: 
    def __init__(self, content, docID):
        self.content = content
        self.docID = docID
        self.term_freq = self.term_frequencies()

    def get_content(self):
        return self.content

    def get_tf(self):
        return self.term_freq

    def get_docID(self):
        return self.docID
    
    def term_frequencies(self) -> Dict[str, int]: 
        frequencies = {}
        for token in self.content.split():
            frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

class Corpus: 
    def __init__(self):
        self.documents = []
        self.doc_freqs = {}
        self.idf = {}
        self.avg_doc_length = 0

    def get_docs(self):
        return self.documents
    
    def get_df(self):
        return self.doc_freqs

    def add_document(self, document: Document): 
        """Add a document to the corpus and update frequencies."""
        self.documents.append(document)
        self.update()

    def avg_doc_len(self):
        total = sum(len(doc.get_content().split()) for doc in self.documents)
        return total / len(self.documents) if self.documents else 0
    
    def df_(self):
        """Calculate and update document frequency for each term."""
        df = {}
        for doc in self.documents:
            for word in set(doc.get_tf()):
                df[word] = df.get(word, 0) + 1
        self.doc_freqs = df

    def idf_(self):
        """Calculate and update inverse document frequency for each term."""
        idf = {}
        corpus_len = len(self.documents)
        for term, freq in self.doc_freqs.items():
            idf[term] = round(math.log((corpus_len + 1) / (freq + 1)), 2)
        self.idf = idf

    def update(self):
        self.df_()
        self.idf_()
        self.avg_doc_length = self.avg_doc_len()
