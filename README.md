Persistent Local AI Document Q\&A Assistant



A Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents, store semantic embeddings in a persistent ChromaDB vector database and ask natural-language questions using a locally hosted Mistral large language model.



The application extracts text from PDF files, creates overlapping document chunks, generates semantic embeddings with Sentence Transformers and performs similarity-based retrieval using ChromaDB. Relevant document passages are supplied to a local Mistral model through Ollama, allowing the system to generate grounded answers supported by source citations and retrieved evidence.



Key Features

PDF document processing and page-aware text extraction

Semantic search using vector embeddings

Persistent ChromaDB vector database

Retrieval-Augmented Generation (RAG)

Local LLM inference with Ollama and Mistral

Source attribution with document and page references

Anti-hallucination testing and validation

Downloadable answer reports

Streamlit-based user interface

Technologies

Python

Streamlit

ChromaDB

Sentence Transformers

all-MiniLM-L6-v2

Ollama

Mistral

pypdf

Retrieval-Augmented Generation (RAG)

Semantic Search

Vector Embeddings

Natural Language Processing (NLP)

Skills Demonstrated

Generative AI Development

Prompt Engineering

Local LLM Deployment

Vector Database Implementation

Semantic Retrieval

Document Intelligence

AI Application Development

Responsible AI Testing

Python Software Development

