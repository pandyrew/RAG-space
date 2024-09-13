from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class FAISSRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # initialize the sentence transformer model and faiss index
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def add_documents(self, documents):
        # add new documents to the retriever and update the faiss index
        self.documents.extend(documents)
        embeddings = self.model.encode(documents)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))

    def retrieve(self, query, k=5):
        # retrieve the k most similar documents to the query
        query_vector = self.model.encode([query])[0]
        distances, indices = self.index.search(np.array([query_vector]).astype('float32'), k)
        return [self.documents[i].strip() for i in indices[0]]