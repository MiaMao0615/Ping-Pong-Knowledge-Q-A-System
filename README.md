# Ping Pong Knowledge Q&A System

## Overview

This project implements a domain-specific question answering system focused on table tennis (ping pong). By integrating a retrieval‑augmented generative pipeline, it answers queries about the sport’s rules, history, techniques, and tournament formats in Chinese. The system uses a hybrid retrieval strategy with dense vector search and sparse lexical search, a BGE (Beijing Academy of AI) re‑ranker for improved relevance, and a large language model (DeepSeek) for natural language generation.

## Installation

1. Ensure **Python 3.8+** is installed.
2. Install required dependencies. This project uses Flask, Flask‑CORS, TongAgents, pymilvus, requests, and Tencent Cloud SDK. You can install them via pip:

   ```bash
   pip install flask flask-cors requests pymilvus tongagents tencentcloud-sdk-python
