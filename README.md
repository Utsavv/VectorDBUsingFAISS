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

## Step-by-Step Breakdown

### Step 0: Setup
To get started, you need to set up your Python environment. Here's a list of dependencies you'll need to install:
```bash
conda install pytorch::faiss-cpu
conda install conda-forge::sentence-transformers
```
### Step 1: Reading the Document File

The first step is reading the documents from a text file. In this solution, each line in the file represents a single document, which simplifies processing and indexing. If your input format differs (e.g., multiple paragraphs per line or JSON format), you may need to adapt the code to appropriately parse and extract individual documents for indexing. For this solution, we assume each line in the file represents a document.

```python
file_path = r'.\Documentation.txt'
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
