import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Step 1: Read the content from the text file

# Step 2: Initialize the embedding model

class EmbeddingModel:
    _model_instance = None

    def __init__(self, document_path, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.document_path = document_path
        self.model_name = model_name
        self.texts = self.load_texts()

    def load_texts(self):
        with open(self.document_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file if line.strip()]

    @classmethod
    def get_model(cls, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        if cls._model_instance is None:
            cls._model_instance = SentenceTransformer(model_name)
        return cls._model_instance

# Step 3: Generate embeddings for each text in batches
file_path = r'.\Azure-AI-Studio-Example\Documentation.txt'
embedding_model = EmbeddingModel(file_path)
texts = embedding_model.texts
model = EmbeddingModel.get_model(embedding_model.model_name)

# Batch size for encoding
batch_size = 32
embeddings = []
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    batch_embeddings = model.encode(batch_texts)
    embeddings.extend(batch_embeddings)

# Step 4: Convert embeddings to numpy array
embeddings_np = np.array(embeddings).astype('float32')  # FAISS requires float32 format

# Step 5: Set up FAISS index
embedding_dim = embeddings_np.shape[1]  # Dimensionality of embeddings
nlist = 100  # Number of clusters
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(embedding_dim), embedding_dim, nlist)  # IVF for better performance on large-scale searches

# Train the index (IVF requires training)
if not index.is_trained:
    index.train(embeddings_np)

# Step 6: Add embeddings to the FAISS index
index.add(embeddings_np)
print(f"Indexed {index.ntotal} documents into FAISS")

# Step 7: Save the index to disk (optional, for persistence)
faiss.write_index(index, 'faiss_index.bin')

# Step 8: Define a class to manage FAISS index loading
class FaissIndex:
    _index_instance = None

    @classmethod
    def get_index(cls, index_path='faiss_index.bin'):
        if cls._index_instance is None:
            if os.path.exists(index_path):
                cls._index_instance = faiss.read_index(index_path)
                print("FAISS index loaded successfully.")
            else:
                cls._index_instance = faiss.IndexIVFFlat(faiss.IndexFlatL2(embedding_dim), embedding_dim, nlist)
                if not cls._index_instance.is_trained:
                    cls._index_instance.train(embeddings_np)
                cls._index_instance.add(embeddings_np)
                print("FAISS index created and loaded successfully.")
        return cls._index_instance

# Step 9: Define the FAISS search function
def search_faiss_index(query):
    # Load model and index
    model = EmbeddingModel.get_model()
    index = FaissIndex.get_index('faiss_index.bin')

    # Handle empty query case
    if not query.strip():
        return "Query is empty. Please provide a valid query."

    # Encode the query
    query_embedding = model.encode([query]).astype('float32')

    # Ensure k is not greater than the number of indexed embeddings
    k = 3  # Number of nearest neighbors to retrieve
    k = min(k, index.ntotal)

    # Search in the index
    distances, indices = index.search(query_embedding, k)

    # Prepare the context from retrieved texts
    retrieved_texts = []
    for idx in indices[0]:
        if 0 <= idx < len(texts):  # Check if the index is within the valid range
            retrieved_texts.append(texts[idx])
    
    answer=""

    # Join the retrieved texts to create context
    if retrieved_texts:
        answer = "\n".join(retrieved_texts)
    else:
        answer = "No relevant information found."

    return answer
