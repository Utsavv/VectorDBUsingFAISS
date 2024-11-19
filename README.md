---
title: Leveraging FAISS and Sentence Embeddings for Document Search
description: Implement an efficient document search system using FAISS and sentence embeddings, useful in applications like chatbots, document retrieval, and natural language understanding.
author: Utsav Verma
date: 2024-11-18
---

# Leveraging FAISS and Sentence Embeddings for Document Search

Searching for relevant information in vast repositories of unstructured text can be a challenge. This article explains a Python-based approach to implementing an efficient document search system using FAISS (Facebook AI Similarity Search) and sentence embeddings, which can be useful in applications like chatbots, document retrieval, and natural language understanding.

In this guide, we will break down how to use FAISS in combination with sentence transformers to create a semantic search solution that can effectively locate related documents based on a user query. For example, this could be used in a customer support system to find the most relevant past tickets or knowledge base articles in response to a user's question.

## Assumptions
This article will focus on explaining and implementing embeddings and vector databases. I am assuming that you have a basic understanding of Python, RAG (Retrieval-Augmented Generation), LLM (Large Language Models).

## What is Embedding?
Embeddings are like special codes that turn words into numbers. Think of words as different puzzle pieces, and embeddings are like a map that shows where each piece fits best. When words mean almost the same thing, their embeddings are like pieces that fit together snugly. This helps computers understand not just what words say, but what they really mean when we use them in sentences.

For example, let's take the sentence 'The cat chased the mouse.' Each word in this sentence, like 'cat' and 'mouse,' gets transformed into a set of numbers that describe its meaning. These numbers help a computer quickly find sentences with similar meanings, like 'The dog chased the rat,' even if the words are different.

Vector databases store these numbers (embeddings) in an efficient way. For instance, in our example sentence 'The cat chased the mouse,' each word ('cat', 'chased', 'mouse') would have its meaning translated into numbers by a computer. These numbers are then organized in a special database that makes it easy for the computer to quickly find similar meanings, like in the sentence 'The dog chased the rat,' even if different words are used.

### Why Understanding Embeddings and Vector Databases Matters for Implementing RAG
In many cases where chatbots use cool AI tricks, like making conversations sound nice, the real magic comes from how we handle all the information. The big challenge is making sure we can find the right information quickly, not just relying on AI alone. Understanding and implementing vector DB is crucial for successful RAG implementations.

## Overview of the Components

Our solution is composed of several major components:

1. **Sentence Transformers for Embeddings**: We use a pre-trained model from the `sentence-transformers` library to convert textual documents into numerical representations (embeddings).
2. **FAISS for Similarity Search**: FAISS, developed by Facebook AI, is used to index these embeddings and perform fast similarity searches on them. This is particularly useful when dealing with large numbers of documents. 

Other options beyond FAISS include **Annoy** (by Spotify), **ScaNN** (by Google), and **HNSWlib**. Each of these libraries offers unique benefits. For instance, Annoy is known for its simplicity and speed, while ScaNN is optimized for Google-scale workloads, and HNSWlib provides excellent accuracy due to its hierarchical navigable small world graphs.

### Advantages and Disadvantages of FAISS

Understanding the advantages and disadvantages of FAISS is crucial for determining if it is the right solution for your needs, especially when considering factors like scalability, performance, and ease of implementation.

- **Advantages**: FAISS is highly optimized for both CPU and GPU, making it capable of handling extremely large datasets efficiently. It supports multiple index types, which provides flexibility for different use cases. Its scalability is particularly suitable for enterprise-level solutions.

- **Disadvantages**: Compared to other solutions, FAISS may require more configuration and tuning to achieve optimal results, and its memory consumption can be relatively high, especially for large datasets. Additionally, setting up GPU acceleration can be complex for some users. Other options like HNSWlib might offer easier setup and competitive performance in some scenarios.

3. **Batched Embedding Generation**: We process the documents in manageable batches to reduce memory consumption and improve the performance of encoding large datasets.

4. **Persistence for Reusability**: The FAISS index is saved to disk for reuse, allowing for quicker lookups in subsequent searches.

### Different Types of Indexes Supported by FAISS

| Name of Index    | Explanation                                                         | Advantages                                     | Disadvantages                                                |
|------------------|---------------------------------------------------------------------|------------------------------------------------|--------------------------------------------------------------|
| **IndexFlatL2**  | A flat (brute-force) index that computes exact distances.           | Simple to implement and exact results.         | Not scalable for large datasets due to linear time queries.  |
| **IndexIVFFlat** | Inverted file with flat vectors, good for large datasets.           | Faster search times on large datasets.         | Requires training and may reduce accuracy slightly.          |
| **IndexIVFPQ**   | Combines IVF with Product Quantization for compression.             | Reduced memory usage and faster searches.      | More complex configuration and can affect recall.            |
| **HNSW**         | Hierarchical Navigable Small World graph for quick searches.        | High accuracy and fast retrieval.              | Memory intensive, and building the index can be slow.        |
| **IndexPQ**      | Uses Product Quantization without IVF, ideal when memory usage is a primary concern. Offers good search performance. | Low memory consumption.                        | Slower compared to other indexing methods with IVF.          |

To summarize, **IndexFlatL2** is best for smaller datasets due to its simplicity, while **IndexIVFFlat** and **IndexIVFPQ** are more suitable for medium to large datasets, providing a good balance between speed and memory usage. **HNSW** is ideal for scenarios requiring high accuracy and fast retrieval, whereas **IndexPQ** is useful when minimizing memory consumption is the primary concern.

## POC Objective
So far, we have covered theoretical part. Given below section demonstrates how to search for queries within a text file.

In production applications, documentation is often extensive and finding information related to a specific topic can be challenging due to scattered information across various documents. To mimic this scenario, I have created a documentation text file. This guide will show you how to search for information within this file. Although a simple text file is used here, the same approach can be applied to PDFs as well.

To make this example more realistic, I used the SAP rule engine documentation available at [SAP Help Portal](https://help.sap.com/docs/SAP_COMMERCE/9d346683b0084da2938be8a285c0c27a/ba076fa614e549309578fba7159fe628.html) and compiled it into a single documentation text file. The text file used in this demonstration is attached to the article and can also be found in the GitHub repository.
## How It Works
The system works in the following steps:

- Load text documents.
- Convert documents into vector embeddings using a Sentence Transformer model.
- Store these embeddings in a FAISS index for efficient similarity search.
- Query the index with user input to retrieve the most relevant documents.

## Step-by-Step Breakdown

### Step 0: Setup
To get started, you need to set up your Python environment. Here's a list of dependencies you'll need to install:
```bash
conda install pytorch::faiss-cpu
conda install conda-forge::sentence-transformers
conda install numpy
```
### 1. Importing the Required Libraries
``` python
import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer
```

**faiss**: The core library used for similarity search. FAISS enables efficient searching through large vector spaces.

**os**: This module is used to interact with the file system, such as listing files in a directory.

**numpy**: Used for handling vector operations and converting embeddings to numerical arrays.

**sentence_transformers**: Provides pre-trained models to convert sentences into dense vector embeddings. These embeddings are used to determine semantic similarity between sentences.

### 2. Defining the Embedding Model and Document Loader Class
``` python
class EmbeddingModel:
    def __init__(self, document_path, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.document_path = document_path
        self.model_name = model_name
        self.model = self.get_model(model_name)
``` 
The EmbeddingModel class initializes with two main attributes:

**document_path**: Path to the documents that need to be processed.

**model_name**: Specifies which pre-trained model to use. Here, it uses the all-MiniLM-L6-v2 model from sentence transformers.

**self.model** loads the pre-trained embedding model using the get_model method.

### 3. Loading Text Documents
``` python
    def load_texts(self):
        texts = []
        for filename in os.listdir(self.document_path):
            with open(os.path.join(self.document_path, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
        return texts
``` 
load_texts reads all documents in the specified document_path directory. It iterates over each file and appends the content to a list called texts.

The utf-8 encoding ensures compatibility with a wide range of text formats.

The method returns the list of texts, which will be used later for embedding generation.

### 4. Model Loading Method
``` python
    @classmethod
    def get_model(cls, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        return SentenceTransformer(model_name)
```
get_model is a class method used to load the specified pre-trained embedding model.

It uses SentenceTransformer from the sentence_transformers library to get the model instance. The embedding model turns text into numerical vectors, which are crucial for similarity search.

### 5. Embedding Generation
``` python
    def generate_embeddings(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)
```
generate_embeddings takes in a list of texts and converts each text into a dense vector representation.

The encode function from the SentenceTransformer model converts text into numerical embeddings.Setting convert_to_numpy=True allows easy use of embeddings in FAISS and Numpy.

### 6. Creating and Training the FAISS Index
``` python
    def create_faiss_index(self, embeddings_np, embedding_dim, nlist=10):
        quantizer = faiss.IndexFlatL2(embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)
        index.train(embeddings_np)
        index.add(embeddings_np)
        return index
``` 
**create_faiss_index** builds an Index for fast similarity search:

**embedding_dim** is the dimensionality of the embedding vectors.

**nlist** is the number of clusters for partitioning the dataset during search.

**IndexFlatL2** is a simple index that computes L2 distances.

**IndexIVFFlat** is used for faster searching by clustering the embeddings.

**train(embeddings_np)** prepares the FAISS index to handle the vector space represented by the embeddings.

**add(embeddings_np)** adds all vectors to the index for similarity search.

### 7. Saving the FAISS Index (Optional)
``` python
    def save_index(self, index, index_path='faiss_index.bin'):
        faiss.write_index(index, index_path)
```
This method saves the trained FAISS index to a file (faiss_index.bin) for later use, which can speed up future searches.

### 8. Loading or Creating FAISS Index Dynamically
``` python
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
``` 
This is a singleton class that ensures only one instance of the FAISS index is loaded.

The get_index method checks if a saved index exists and loads it. If an index does not exist, it creates and trains a new one, then adds embeddings.

### 9. Defining the Search Function
``` python
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
        if 0 <= idx < len(texts):
            retrieved_texts.append(texts[idx])
    
    answer=""

    # Join the retrieved texts to create context
    if retrieved_texts:
        answer = "\n".join(retrieved_texts)
    else:
        answer = "No relevant information found."

    return answer
```
**search_faiss_index** allows users to input a text query: It first encodes the query using the pre-trained embedding model.

**index.search(query_embedding, k)** finds the k most similar entries in the FAISS index. It Retrieves the corresponding documents based on similarity scores. It Returns a combined result of relevant documents or an appropriate message if no matches are found.

### Running the Script

**Prepare Documents**: Place all text documents in a folder. Specify the folder path when initializing the EmbeddingModel.

**Generate Embeddings**: Use EmbeddingModel to load texts and generate embeddings.

**Create and Train FAISS Index**: Pass the embeddings to create_faiss_index() to build your FAISS index.

**Search**: Use search_faiss_index(query) to find the most relevant documents for your query.

### Example Usage

####  Initialize the embedding model
``` python
embedding_model = EmbeddingModel(document_path='path/to/documents')
``` 
#### Load and encode the documents
``` python
doc_texts = embedding_model.load_texts()
embeddings_np = embedding_model.generate_embeddings(doc_texts)
``` 
#### Create and train the FAISS index
``` python
faiss_index = embedding_model.create_faiss_index(embeddings_np, embedding_dim=embeddings_np.shape[1])
``` 
#### Save the FAISS index
``` python
embedding_model.save_index(faiss_index)
``` 
#### Perform a search
``` python
result = search_faiss_index("sample query")
print(result)
``` 

## Conclusion

This solution provides a scalable approach to searching large volumes of text efficiently by combining sentence embeddings and FAISS. It highlights the power of semantic search over simple keyword matching by considering the meaning of the query in finding related documents.

Using FAISS and sentence transformers together allows us to handle large datasets with good performance, providing relevant results to user queries. Such a setup is especially beneficial for applications involving document retrieval, chatbots, and any solution requiring similarity-based matching.

We encourage readers to experiment with different embedding models and FAISS index types to optimize the solution for specific use cases. Additionally, feel free to modify the batch sizes or index parameters to best suit the characteristics of your dataset.
