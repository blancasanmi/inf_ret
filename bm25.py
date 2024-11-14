import math
from bm25 import Corpus, Document

class BM25Scorer: 
    def __init__(self, corpus: Corpus, k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b

    def _score(query, doc : Document, docs : Corpus, k1=1.5, b=0.75):
        score = 0.0
        tf = doc.get_tf()
        df = docs.get_df()
        idf = docs.get_idf()
        avg_doc_len = docs.avg_doc_len()

        for term in query:
            if term not in tf.keys():
                continue
            score += math.log(idf)*((k1 + 1)*tf[term])/(k1*((1-b)+b*(len(doc)/avg_doc_len))+tf[term])
        return score
    
class BM25Retriever: 
    def __init__(self, corpus: Corpus, scorer: BM25Scorer):
        self.corpus = corpus
        self.scorer = scorer
        self.ranking = {}

    def add_score(self, docID, score):
        self.ranking[docID] = score

    def sort_ranking(self):
        self.ranking.sort(key=lambda x: x[1], reverse=True)

    def retrieve(self, query, top_n):
        self.ranking = []  # Reset ranking for each query

        for doc in self.corpus: 
            self.add_score(doc.get_docID(), self.scorer._score(query, doc, self.corpus))

        self.sort_ranking()

        # Return the top N documents
        return self.ranking[:top_n]

        
            
