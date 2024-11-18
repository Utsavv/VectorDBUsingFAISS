---
title: Leveraging FAISS and Sentence Embeddings for Document Search
description: Implement an efficient document search system using FAISS and sentence embeddings, useful in applications like chatbots, document retrieval, and natural language understanding.
author: Utsav Verma
date: 2024-11-18
---

# Leveraging FAISS and Sentence Embeddings for Document Search

Searching for relevant information in vast repositories of unstructured text can be a challenge. This article explains a Python-based approach to implementing an efficient document search system using FAISS (Facebook AI Similarity Search) and sentence embeddings, which can be useful in applications like chatbots, document retrieval, and natural language understanding.

In this guide, we will break down how to use FAISS in combination with sentence transformers to create a semantic search solution that can effectively locate related documents based on a user query. For example, this could be used in a customer support system to find the most relevant past tickets or knowledge base articles in response to a user's question.

##Assumptions
This article will focus on implementing embeddings. I am assuming that you have basic understanding of python, RAG, LLM.

##What is Embedding?
Embedding introduction

### Why understanding Embeddings and Vector DB important for RAG implementation?
In many production RAG implementation scenario like chatbot implementation, LLM/AI works like a beautifier only. Real magic happens at your data. Main challange is in searching from your data not in using LLM.

## Overview of the Components

Our solution is composed of several major components:

1. **Sentence Transformers for Embeddings**: We use a pre-trained model from the `sentence-transformers` library to convert textual documents into numerical representations (embeddings).
2. **FAISS for Similarity Search**: FAISS, developed by Facebook AI, is used to index these embeddings and perform fast similarity searches on them. This is particularly useful when dealing with large numbers of documents. Other options beyond FAISS include **Annoy** (by Spotify), **ScaNN** (by Google), and **HNSWlib**. Each of these libraries offers unique benefits. For instance, Annoy is known for its simplicity and speed, while ScaNN is optimized for Google-scale workloads, and HNSWlib provides excellent accuracy due to its hierarchical navigable small world graphs.

### Advantages and Disadvantages of FAISS

Understanding the advantages and disadvantages of FAISS is crucial for determining if it is the right solution for your needs, especially when considering factors like scalability, performance, and ease of implementation.

- **Advantages**: FAISS is highly optimized for both CPU and GPU, making it capable of handling extremely large datasets efficiently. It supports multiple index types, which provides flexibility for different use cases. Its scalability is particularly suitable for enterprise-level solutions.
- **Disadvantages**: Compared to other solutions, FAISS may require more configuration and tuning to achieve optimal results, and its memory consumption can be relatively high, especially for large datasets. Additionally, setting up GPU acceleration can be complex for some users. Other options like HNSWlib might offer easier setup and competitive performance in some scenarios.

3. **Batched Embedding Generation**: We process the documents in manageable batches to reduce memory consumption and improve the performance of encoding large datasets.

4. **Persistence for Reusability**: The FAISS index is saved to disk for reuse, allowing for quicker lookups in subsequent searches.

5. The code implementation follows these steps, and we'll now explore each part in detail.


### Different Types of Indexes Supported by FAISS

| Name of Index    | Explanation                                                                                                                                        | Advantage                                        | Disadvantage                                                       |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------ |
| **IndexFlatL2**  | Performs a brute-force search over all vectors. Suitable for small datasets where search speed is not a concern.                                   | Simple and effective for small datasets.         | Not efficient for large datasets due to high computational cost.   |
| **IndexIVFFlat** | Uses Inverted File (IVF) lists to cluster vectors, allowing for faster searches. Ideal for medium to large datasets where search time is critical. | Faster searches compared to brute-force methods. | Requires training and may not be as accurate as exhaustive search. |
| **IndexIVFPQ**   | Utilizes Product Quantization (PQ) to reduce memory footprint. Balances speed and memory usage, suitable for very large datasets.                  | Efficient in terms of both speed and memory.     | Lower accuracy due to quantization.                                |
| **HNSW**         | Designed for fast, high-accuracy searches using a graph-based approach. Best for scenarios requiring higher recall.                                | High accuracy and fast retrieval.                | Memory intensive, and building the index can be slow.              |
| **IndexPQ**      | Uses Product Quantization without IVF, ideal when memory usage is a primary concern. Offers good search performance.                               | Low memory consumption.                          | Slower compared to other indexing methods with IVF.                |

To summarize, **IndexFlatL2** is best for smaller datasets due to its simplicity, while **IndexIVFFlat** and **IndexIVFPQ** are more suitable for medium to large datasets, providing a good balance between speed and memory usage. **HNSW** is ideal for scenarios requiring high accuracy and fast retrieval, whereas **IndexPQ** is useful when minimizing memory consumption is the primary concern.


## Step-by-Step Breakdown

### Step 1: Reading the Document File

The first step is reading the documents from a text file. In this solution, each line in the file represents a single document, which simplifies processing and indexing. If your input format differs (e.g., multiple paragraphs per line or JSON format), you may need to adapt the code to appropriately parse and extract individual documents for indexing. For this solution, we assume each line in the file represents a document.

```python
file_path = r'.\Azure-AI-Studio-Example\Documentation.txt'
embedding_model = EmbeddingModel(file_path)
texts = embedding_model.texts
```

This step reads the contents of `Documentation.txt` and stores each line as an entry in the list `texts`.

### Step 2: Embedding Model Initialization

We use `sentence-transformers` to initialize our embedding model. Embedding models convert textual data into a vector format that can be used for similarity comparison.

```python
model = EmbeddingModel.get_model(embedding_model.model_name)
```

The model used here is `'sentence-transformers/all-MiniLM-L6-v2'`, but this is configurable, allowing flexibility for different use cases.

### Step 3: Generate Embeddings in Batches

Converting all text documents into embeddings at once might lead to high memory usage, especially with large datasets. Therefore, we generate embeddings in manageable batches:

```python
batch_size = 32
embeddings = []
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    batch_embeddings = model.encode(batch_texts)
    embeddings.extend(batch_embeddings)
```

This approach reduces memory pressure, ensuring that the solution scales well even for larger files.

### Step 4: Convert Embeddings to Numpy Array

FAISS requires embeddings to be in a specific format (`float32`). We convert our embeddings to the required format:

```python
embeddings_np = np.array(embeddings).astype('float32')
```

### Step 5: Set Up FAISS Index

We use FAISS to create an index for the embeddings. In this example, we use `IndexIVFFlat`, which performs clustering to speed up searches on large datasets:

```python
embedding_dim = embeddings_np.shape[1]
nlist = 100
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(embedding_dim), embedding_dim, nlist)
```

The `nlist` parameter defines the number of clusters to use, which affects the search speed and accuracy. After setting up the index, we train it with the generated embeddings.

### Step 6: Adding Embeddings to the Index

Once the index is trained, we add our embeddings to it:

```python
index.add(embeddings_np)
print(f"Indexed {index.ntotal} documents into FAISS")
```

This allows us to later query the index for similar documents based on a given input.

### Step 7: Saving the FAISS Index

To avoid retraining and adding embeddings each time, we save the index to disk:

```python
faiss.write_index(index, 'faiss_index.bin')
```

This saved index can be reloaded for future searches, making the solution more efficient.

### Step 8: Manage FAISS Index Loading

We define a class to manage the loading of the FAISS index, so that we either load an existing index from disk or create a new one if it doesn't exist:

```python
class FaissIndex:
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

This ensures that the index is always ready to be used when searching.

### Step 9: Search the FAISS Index

Finally, we implement the function to search the FAISS index. The retrieved results can be used in broader applications, such as populating a user interface with relevant information, further processing for analytics, or providing context-aware responses in a chatbot. Given a query, we encode it and search for the closest matches in the index:

```python
def search_faiss_index(query):
    model = EmbeddingModel.get_model()
    index = FaissIndex.get_index('faiss_index.bin')

    if not query.strip():
        return "Query is empty. Please provide a valid query."

    query_embedding = model.encode([query]).astype('float32')
    k = min(3, index.ntotal)
    distances, indices = index.search(query_embedding, k)

    retrieved_texts = [texts[idx] for idx in indices[0] if 0 <= idx < len(texts)]
    return "\n".join(retrieved_texts) if retrieved_texts else "No relevant information found."
```

The search function checks for empty queries, encodes the input, and retrieves the nearest documents from the FAISS index. It also ensures `k` (number of results) is not greater than the number of indexed documents, avoiding errors.

## Conclusion

This solution provides a scalable approach to searching large volumes of text efficiently by combining sentence embeddings and FAISS. It highlights the power of semantic search over simple keyword matching by considering the meaning of the query in finding related documents.

Using FAISS and sentence transformers together allows us to handle large datasets with good performance, providing relevant results to user queries. Such a setup is especially beneficial for applications involving document retrieval, chatbots, and any solution requiring similarity-based matching.

We encourage readers to experiment with different embedding models and FAISS index types to optimize the solution for specific use cases. Additionally, feel free to modify the batch sizes or index parameters to best suit the characteristics of your dataset.

