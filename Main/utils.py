"""
Utilities for the ping pong knowledge agent.

This module defines helper classes and functions used to connect to remote
embedding services, reranking services, and a hybrid search engine built on
Milvus. These utilities are leveraged by the table tennis agent to embed
queries and documents, perform dense and sparse vector searches, rerank
results, and manage data stored in the knowledge base. All comments and
docstrings in this file are written in English.
"""

import os
import json
from tests.models import ModelStore
from tongagents.knowledge.vectorstore.milvus import (
    MilvusBGEHybridSearchEngine,
)
from tongagents.knowledge.embedding.bge_embedding import BGEM3Embedding

import logging

from pymilvus import (
    AnnSearchRequest,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)

from tongagents.knowledge.core import (
    KnowledgeStoreInfo,
    SearchResult,
)
from tongagents.knowledge.embedding.bge_embedding import (
    BGEM3Embedding,
    LocalBGEM3Embedding,
)

log = logging.getLogger(__name__)

from typing import Literal

import numpy as np
import requests

MODEL_STORE = ModelStore()
COLLECTION_NAME = "test_milvus_collection"

class RemoteBGEM3EmbeddingClient(BGEM3Embedding):
    def __init__(self, embedding_endpoint: str, timeout=10):
        self.embedding_endpoint = embedding_endpoint
        self.timeout = timeout
        self.dimensions = 1024

    def embed_texts(
        self, texts: list[str]
    ) -> dict[
        Literal["dense_vecs", "lexical_weights", "colbert_vecs"],
        np.ndarray | list[dict[str, float]] | list[np.ndarray],
    ]:
        response = requests.post(
            self.embedding_endpoint, json=texts, timeout=self.timeout
        )
        if response.status_code != 200:  # noqa: PLR2004
            raise ValueError(
                f"Failed to request bge-m3 embedding on url {self.embedding_endpoint},\
                status code: {response.status_code}"
            )
        return response.json()

class RemoteBGERerankerClient:
    def __init__(self, rerank_endpoint: str, timeout=10):
        """
        :param rerank_endpoint: URL of the remote rerank service, e.g., "http://localhost:8000/rerank"
        :param timeout: request timeout in seconds
        """
        self.rerank_endpoint = rerank_endpoint
        self.timeout = timeout

    def rerank(
        self, query: str, documents: list[str], top_k: int = 10
    ) -> list[dict[str, float | str]]:
        """
        Call the remote rerank service and return the document ranking results with scores
        """
        payload = {
            "query": query,
            "documents": documents,
            "limit": top_k
        }
        response = requests.post(
            self.rerank_endpoint, json=payload, timeout=self.timeout
        )

        if response.status_code != 200:
            raise ValueError(
                f"Failed to rerank on url {self.rerank_endpoint}, "
                f"status code: {response.status_code}, detail: {response.text}"
            )

        return response.json()

class NewMilvusBGEHybridSearchEngine(MilvusBGEHybridSearchEngine):
    def __init__(
        self,
        collection_name="default_hybrid_collection",
        uri="./local_milvus.db",  # "http://localhost:19530",
        bge_embedding_model: BGEM3Embedding | None = None,
        bge_reranker_model=None,
        model_name="BAAI/bge-m3",
        overwrite=False,
        index_type=None,
        metric_type=None,
        search_limit=None,
    ):
        super().__init__(collection_name, uri, bge_embedding_model, model_name, overwrite, index_type, metric_type, search_limit)
        # pylint: disable=import-outside-toplevel
        from pymilvus.model.reranker import BGERerankFunction

        if not bge_reranker_model:
            bge_reranker_model = BGERerankFunction(
                model_name="BAAI/bge-reranker-v2-m3",  # Specify the model name. Defaults to `BAAI/bge-reranker-v2-m3`.
                device="cuda:0" # Specify the device to use, e.g., 'cpu' or 'cuda:0'
            )
        self.ranker = bge_reranker_model

    def search(
        self, query: str, limit: int | None = None, **kwargs
    ) -> list[SearchResult]:
        embeddings = self.bge_embedding_model([query])
        limit = limit or self.search_limit
        
        # Search in the dense vector space
        dense_req = AnnSearchRequest(
            embeddings["dense_vecs"],
            "dense_vector",
            {"metric_type": self.metric_type},
            limit=limit,
        )
        dense_results = self.milvus_client.search(
            self.collection_name,
            data=[dense_req.data[0]],
            anns_field=dense_req.anns_field,
            output_fields=["id", "content", "source", "page"],
            limit=limit,
        )[0]
        
        # Search in the sparse vector space
        sparse_req = AnnSearchRequest(
            embeddings["lexical_weights"],
            "sparse_vector",
            {"metric_type": "IP"},
            limit=limit,
        )
        sparse_results = self.milvus_client.search(
            self.collection_name,
            data=[sparse_req.data[0]],
            anns_field=sparse_req.anns_field,
            output_fields=["id", "content", "source", "page"],
            limit=limit,
        )[0]
        
        # Extract a unique result list
        unique_results = list(dense_results + sparse_results)
        
        # Prepare a list of documents for reranking
        documents = [result["content"] for result in unique_results]
        
        # Use the BGE reranker to reorder the results
        reranked_results = self.ranker.rerank(query=query, documents=documents, top_k=min(limit, len(documents)))
                
        # Create the final result list
        final_results = []
        for reranked_result in reranked_results:
            idx = reranked_result['index']
            original_result = unique_results[idx]
            original_result["score"] = reranked_result['score']
            final_results.append(original_result)
        
        # Convert to SearchResult format
        docs = [self._convert_to_query_result(r) for r in final_results]
        return docs  

def load_json(file_name):
    return json.load(open(file_name, "r"))

def load_jsonl(file_name):
    data_list = []
    with open(file_name, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            data_list.append(json.loads(line))
    return data_list

def write_to_json(data, file_name, indent=2):
    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=indent)

def write_to_jsonl(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as jsonl_file:
        for item in data:
            jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')

def get_dir_data(dir):
    case_list = []
    for file_name in os.listdir(dir):
        case_list += load_json(f"{dir}/{file_name}")
    return case_list

def get_store(name, url, model_name='BAAI_bge-m3'):
    return NewMilvusBGEHybridSearchEngine(
        collection_name=name,
        model_name=MODEL_STORE.get_model(model_name),
        uri=url,
        bge_embedding_model=RemoteBGEM3EmbeddingClient(embedding_endpoint="http://10.1.53.171/v1/bgem3/embedding"),
        bge_reranker_model=RemoteBGERerankerClient(rerank_endpoint="http://localhost:8002/rerank"),
    )