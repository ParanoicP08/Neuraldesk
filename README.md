# NeuralDesk

A RAG app that runs entirely in the browser — no backend, no API costs.

## What it does
Upload any document and ask questions about it. The app finds the most 
relevant parts and sends them to a local LLM to generate an answer.

## Features
- Document upload (.txt, .md, .csv, .json, .html)
- Built my own vector search in JS using cosine similarity
- Chunk-based retrieval — splits docs into overlapping chunks
- Streams responses token by token from Ollama
- Two modes — RAG (document-aware) and plain Chat

## Stack
- React
- Custom TF vector store (zero external dependencies)
- Ollama + DeepSeek-R1:7b

## How it works
1. Upload a file
2. It gets split into chunks and converted to TF vectors
3. When you ask something, your question gets vectorized too
4. Closest matching chunks get retrieved via cosine similarity
5. Those chunks + your question go to DeepSeek via Ollama
6. Answer streams back in real time

## Why I built this
Wanted to understand how RAG actually works under the hood 
instead of just using a library for it.
