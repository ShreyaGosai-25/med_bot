# MED_BOT

## Table of Contents

- Overview
- Features
- System Architecture
- Key Components
  - Document Processing
  - Vector Database
  - Retrieval-Augmented Generation (RAG)
  - Image Captioning
- UI/UX and Web Deployment
- Installation and Usage
  - Prerequisites
  - Setup
- License

---

## Overview

MedAssist is an intelligent Medical RAG Chatbot designed to answer healthcare-related queries using medical documents and AI-powered retrieval. The system combines semantic search, vector databases, and Large Language Models to provide context-aware responses.

It integrates:

- Medical Document Processing
- Semantic Search with Embeddings
- Retrieval-Augmented Generation (RAG)
- Llama 2 Language Model
- Image Captioning using BLIP

---

## Features

- Medical Question Answering
- PDF-Based Knowledge Retrieval
- Semantic Search using Vector Embeddings
- FAISS & Pinecone Support
- Llama 2 Integration
- Medical Safety Prompting
- Image Captioning Support
- Fast and Interactive Web Interface

---

## System Architecture

Medical PDFs → Text Chunking → Embeddings → Vector Database → Retriever → Llama 2 → Response

Image → BLIP Captioning → Medical Context → Response

---

## Key Components

### 1. Document Processing

Loads and processes medical PDF documents for knowledge extraction.

### 2. Vector Database

Stores document embeddings using FAISS or Pinecone for efficient retrieval.

### 3. Retrieval-Augmented Generation (RAG)

Fetches relevant medical information before generating responses.

### 4. Image Captioning

Uses BLIP to generate captions from uploaded images for multimodal interaction.

---

## UI/UX and Web Deployment

- Simple Chat Interface
- Real-Time Medical Query Response
- Fast Retrieval System
- Image Captioning Support
- Flask-Based Web Application

---

## Installation and Usage

### Prerequisites

- Python 3.10+
- LangChain
- Flask
- Pinecone
- Hugging Face Transformers

### Setup

```bash
git clone https://github.com/your-username/medical-chatbot.git

cd medical-chatbot

python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open:

```text
http://localhost:8080
```

---

## 🚀 Tools We Have Used

Python • Flask • LangChain • Pinecone • FAISS • Hugging Face • Llama 2 • BLIP

---

## License

Licensed under the MIT License. See LICENSE for full details.
