import numpy as np
from langchain_openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document as LangChainDocument
import os

from corpus import Corpus, Document
from retrievers_ import Retriever, Scorer

class FAISSScorer(Scorer):
    def __init__(self, corpus: Corpus):
        super().__init__(corpus)
        self.generate_library()
        
    def set_library(self, lib):
        self.library = lib

    def get_library(self):
        return self.library
    
    def generate_library(self):
        documents = self.corpus.get_docs()
        langchain_documents = [
            LangChainDocument(
                page_content=doc.get_content(),  # Assuming `get_content()` provides the text
                metadata={"docID": doc.docID}   # Include metadata, like docID or any other info
            )
            for doc in documents
        ]

        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

        # Split the LangChain documents into smaller chunks
        split_docs = text_splitter.split_documents(langchain_documents)

        embeddings = OpenAIEmbeddings(openai_api_key="")

        library = FAISS.from_documents(split_docs, embeddings) #equiv of vectorstore

        self.set_library(library)
        print("library should have been saved")

class FAISSRetriever(Retriever):
    def __init__(self, corpus: Corpus, scorer: Scorer):
        super().__init__(corpus, scorer)

    def retrieve(self, query, top_n):
        results_with_scores = self.scorer.library.similarity_search_with_score(query, k=top_n)
        return [
            {
                "content": result.page_content,
                "metadata": result.metadata,
                "score": score,
            }
            for result, score in results_with_scores
        ]
