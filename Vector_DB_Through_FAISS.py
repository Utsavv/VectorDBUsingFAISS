import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer



# Step 1: Read the content from the text file
file_path = r'.\Azure-AI-Studio-Example\Documentation.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    texts = [line.strip() for line in file if line.strip()]

# Step 2: Initialize the embedding model
class EmbeddingModel:
    _model_instance = None

    @classmethod
    def get_model(cls):
        if cls._model_instance is None:
            cls._model_instance = SentenceTransformer('all-MiniLM-L6-v2')  # Load the model only once
            print("Model loaded successfully.")
        return cls._model_instance

# Step 3: Generate embeddings for each text
model = EmbeddingModel.get_model()
embeddings = model.encode(texts)

# Step 4: Convert embeddings to numpy array
embeddings_np = np.array(embeddings).astype('float32')  # FAISS requires float32 format

# Step 5: Set up FAISS index
embedding_dim = embeddings_np.shape[1]  # Dimensionality of embeddings
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity search

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
                cls._index_instance = faiss.IndexFlatL2(embedding_dim)
                cls._index_instance.add(embeddings_np)
                print("FAISS index created and loaded successfully.")
        return cls._index_instance

# Step 9: Define the FAISS search function
def search_faiss_index(query):
    # Load model and index
    model = EmbeddingModel.get_model()
    index = FaissIndex.get_index('faiss_index.bin')

    # Encode the query
    query_embedding = model.encode([query]).astype('float32')

    # Search in the index
    k = 3  # Number of nearest neighbors to retrieve
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