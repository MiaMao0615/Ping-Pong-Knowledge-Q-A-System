# Ping Pong Knowledge Q&A System

## Overview

This project implements a domain-specific question answering system focused on table tennis (ping pong). By integrating a retrieval‑augmented generative pipeline, it answers queries about the sport’s rules, history, techniques, and tournament formats in Chinese. The system uses a hybrid retrieval strategy with dense vector search and sparse lexical search, a BGE (Beijing Academy of AI) re‑ranker for improved relevance, and a large language model (DeepSeek) for natural language generation.

## Features

- **Domain knowledge on table tennis:** answers questions about official match rules, equipment standards, scoring systems, history and major tournaments.
- **Hybrid retrieval pipeline:** first‑stage recall using both dense embeddings (via BGE‑M3) and sparse keyword search in Milvus, followed by a cross‑encoder BGE re‑ranker (`BGE‑Reranker v2 m3`).
- **Query rewriting:** automatically converts incomplete or unclear user questions into clear, self‑contained queries using the DeepSeek API.
- **Query classification:** determines whether a query is related to table tennis based on a configurable keyword list.
- **Context‑aware answering:** maintains conversation history and uses it during query rewriting and answer generation.
- **Streaming responses:** provides server‑sent events (SSE) for incremental delivery of the answer.
- **Configurable knowledge stores:** supports multiple knowledge bases with customizable recall thresholds and sizes.
- **REST API endpoints:** endpoints for chat, streaming, clearing and saving chat history.
- **Modular design:** separate modules handle embedding, re‑ranking, retrieval, LLM integration, and utility functions.

## Architecture

### app.py

The main Flask application orchestrates the chat service. It:

- Loads environment variables `SECRET_ID`, `SECRET_KEY`, and `API_KEY` for Tencent Cloud DeepSeek API authentication.
- Configures a `ReactAgent` from the TongAgents framework with a Chinese system prompt for table tennis and a web search tool.
- Initializes one or more Milvus knowledge stores (for example `test_hybrid` and `tong_knowledge_offline_store`) whose names are exposed as “乒乓球知识库1” and “乒乓球知识库2”.
- Implements functions to rewrite queries, classify their domain relevance, fetch recall results from stores, re‑rank them, and generate final answers using DeepSeek.
- Exposes RESTful endpoints:
  - **`/api/chat`** – synchronous chat via the retrieval pipeline.
  - **`/api/chat_with_api`** – direct call through the agent’s `your_model_function` if you prefer to manage the recall pipeline yourself.
  - **`/stream`** – streaming responses via SSE.
  - **`/api/clear-history`** – clear server‑side session data.
  - **`/api/save-chat`** – save chat history to a JSON file in the `chats/` folder.
  - Serves static front‑end content at `/`.

### utils.py

The utility module provides:

- **`RemoteBGEM3EmbeddingClient`** to fetch BGE‑M3 embeddings from a remote embedding service.
- **`RemoteBGERerankerClient`** to call a remote re‑ranking service.
- **`NewMilvusBGEHybridSearchEngine`**, a subclass of Milvus search engine that performs dense and sparse search and re‑ranks results with the BGE re‑ranker.
- Helper functions for loading and writing JSON/JSONL data.
- **`get_store`** to create a configured hybrid search engine with custom endpoints for embedding and re‑ranking.

### deepseek_api.py

This module wraps the Tencent Cloud DeepSeek API, enabling streaming and non‑streaming completions. It creates and uses a `Credential` object, sends chat completion requests, and yields token streams.

## Installation

1. Ensure **Python 3.8+** is installed.
2. Install required dependencies. This project uses Flask, Flask‑CORS, TongAgents, pymilvus, requests, and Tencent Cloud SDK. You can install them via pip:

   ```bash
   pip install flask flask-cors requests pymilvus tongagents tencentcloud-sdk-python
