from corpus import Corpus, Document

class Scorer: 
    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def score_(self, query, doc: Document) -> float:
         pass


class Retriever: 
    def __init__(self, corpus: Corpus, scorer: Scorer):
        self.corpus = corpus
        self.scorer = scorer
        self.ranking = []
    
    def add_score(self, docID, score):
        self.ranking.append((docID, score))

    def sort_ranking(self):
        self.ranking.sort(key=lambda x: x[1], reverse=True)

    def retrieve(self, query, top_n):
        """Retrieve the top N documents for a given query."""
        self.ranking = []  # Reset ranking for each query

        for doc in self.corpus.get_docs():
            score = self.scorer._score(query, doc)
            self.add_score(doc.get_docID(), score)

        # Sort the ranking by score and return the top N documents
        self.sort_ranking()
        return self.ranking[:top_n]

