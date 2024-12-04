from corpus import Corpus, Document
from bm25 import BM25Retriever, BM25Scorer
from faiss import FAISSRetriever, FAISSScorer
from unigram import UnigramLanguageModel

corpus = Corpus()
corpus.add_document(Document("singapore is in asia", docID=1))
corpus.add_document(Document("the merlion is in singapore", docID=2))
corpus.add_document(Document("independence of singapore was in 1965", docID=3))

query = "where is singapore"

def QandA(retriever, query): 
    results = retriever.retrieve(query, top_n=3)
    print("Top Documents:", results)

QandA(BM25Retriever(corpus, BM25Scorer(corpus)), query)
# QandA(FAISSRetriever(corpus, FAISSScorer(corpus)), query)


## testing out the unigram model 
def plot_top_k_words(unigram_model, k):
    # Sort the unigram model by frequency and get the top k words:
    sorted_words = sorted(unigram_model.items(), key=lambda item: item[1], reverse=True)[:k]

    # Separate the words and their frequencies:
    words, frequencies = zip(*sorted_words)

    # Create the histogram:
    plt.figure(figsize=(12, 8))
    plt.bar(words, frequencies, color='blue')


    plt.title(f"Top {k} Words Frequency Histogram")
    plt.xlabel('Words')
    plt.ylabel('Frequencies')

    plt.xticks(rotation=45)

    plt.show()