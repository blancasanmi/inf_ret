from corpus import Corpus, Document
from retrievers_ import Retriever, Scorer

class BM25Scorer(Scorer): 
    def __init__(self, corpus: Corpus, k1: float = 1.5, b: float = 0.75):
        super().__init__(corpus)
        # self.corpus = corpus
        self.k1 = k1
        self.b = b

    def _score(self, query, doc: Document) -> float:
        """Calculate the BM25 score of a document given a query."""
        score = 0.0
        tf = doc.get_tf()
        df = self.corpus.get_df()
        idf = self.corpus.idf
        avg_doc_len = self.corpus.avg_doc_len()
        doc_len = len(doc.get_content().split())

        for term in query.split():
            if term not in tf:
                continue
            term_idf = idf.get(term, 0)
            term_tf = tf[term]
            score += term_idf * ((self.k1 + 1) * term_tf) / (
                self.k1 * ((1 - self.b) + self.b * (doc_len / avg_doc_len)) + term_tf
            )
        return score
    
class BM25Retriever(Retriever): 
    def __init__(self, corpus: Corpus, scorer: BM25Scorer):
        super().__init__(corpus, scorer)
        self.scorer = scorer
