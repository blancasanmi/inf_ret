from typing import List, Dict, Optional, Tuple

class Document: 
    def __init__(self, content, docID):
        self.content = content
        self.docID = docID
        self.term_freq = self.term_frequencies()
    
    def get_content(self):
        return self.content
    
    # taken from my assignment 4 
    def term_frequencies(self) -> Dict[str, int]: 
        frequencies = {}
        for token in self.get_content():
            if token in frequencies:
                frequencies[token] += 1
            else:
                frequencies[token] = 1
        return frequencies
    
class Corpus: 
    def __init__(self):
        self.documents = []
        self.doc_freqs = {}
        self.avg_doc_length = 0
    
    # getters and setters
    def get_docs(self):
        return self.documents
    
    def set_docs(self, updated):
        self.documents = updated
    
    def set_avg_doc(self, new_avg):
        self.avg_doc_length = new_avg

    def set_df(self, new_freq): 
        self.doc_freqs = new_freq

    # METHODS

    def avg_doc_len(self):
        total = 0
        docs = self.get_docs()
        for doc in docs:
            total += len(doc)
        return total / len(self.get_docs)
    
    def df_(self): # aslo taken from assignment
        df = {}
        docs = self.get_docs()
        for doc in docs:
            for word in doc:
                if word in df:
                    df[word] += 1
                else:
                    df[word] = 1
        # update the document frequency newly calculated
        self.set_df(df)

    def add_doc(self, document: Document): # TODO change document frequencies and length 
        docs = self.get_docs()
        docs.append(document)
        self.set_docs(docs)

        # update everything 
        self.update()       

    def update(self):
        self.df_()
        self.avg_doc_len()