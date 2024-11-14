from corpus import Corpus, Document
from bm25 import BM25Retriever, BM25Scorer

corpus = Corpus()
corpus.add_document(Document("The quick brown fox", doc_id=1))
corpus.add_document(Document("The lazy dog", doc_id=2))
corpus.add_document(Document("The quick dog jumps", doc_id=3))

print(corpus.get_docs())

# scorer = BM25Scorer(corpus)
# retriever = BM25Retriever(corpus, scorer)
# query = "quick fox"
# results = retriever.retrieve(query, top_n=2)
# print("Top Documents:", results)
