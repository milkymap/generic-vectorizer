

import asyncio
import grpc
from typing import List, Dict, AsyncGenerator
from operator import attrgetter
from contextlib import asynccontextmanager
from generic_vectorizer.grpc_server.interfaces.strategies_pb2 import (
    TextEmbeddingRequest, TextEmbeddingResponse,
    TextBatchEmbeddingRequest, TextBatchEmbeddingResponse,
    TextRerankScoresRequest, TextRerankScoresResponse
)
from generic_vectorizer.grpc_server.interfaces import strategies_pb2_grpc
from generic_vectorizer.log import logger 

class AsyncGRPCEmbeddingClient:
    def __init__(self, grpc_server_address: str):
        self.grpc_server_address = grpc_server_address

    @asynccontextmanager
    async def create_grpc_stub(self, stub_type: str = 'TextEmbeddingStub') -> AsyncGenerator[strategies_pb2_grpc.TextEmbeddingStub, None]:
        async with grpc.aio.insecure_channel(target=self.grpc_server_address) as channel:
            try:
                stub = attrgetter(stub_type)(strategies_pb2_grpc)(channel=channel)
                yield stub
            except Exception as e:
                logger.error(e)
                raise

class AsyncEmbeddingClient:
    def __init__(self, grpc_server_address: str):
        self.quick_embed_client = AsyncGRPCEmbeddingClient(grpc_server_address)

    async def get_embedding(self, text: str, target_topic: str, 
                            chunk_size: int = 512, return_dense: bool = True, 
                            return_sparse: bool = False) -> Dict:
        async with self.quick_embed_client.create_grpc_stub() as stub:
            request = TextEmbeddingRequest(
                target_topic=target_topic,
                text=text,
                chunk_size=chunk_size,
                return_dense=return_dense,
                return_sparse=return_sparse
            )
            response: TextEmbeddingResponse = await stub.getTextEmbedding(request)
            if not response.status:
                raise Exception(f"Embedding failed: {response.error}")
            return {
                "dense": list(response.embedding.dense_values) if return_dense else None,
                "sparse": dict(response.embedding.sparse_values) if return_sparse else None
            }

    async def get_batch_embedding(self, texts: List[str], 
                                  target_topic: str, chunk_size: int = 512, 
                                  return_dense: bool = True, return_sparse: bool = False) -> List[Dict]:
        async with self.quick_embed_client.create_grpc_stub() as stub:
            request = TextBatchEmbeddingRequest(
                target_topic=target_topic,
                texts=texts,
                chunk_size=chunk_size,
                return_dense=return_dense,
                return_sparse=return_sparse
            )
            response: TextBatchEmbeddingResponse = await stub.getTextBatchEmbedding(request)
            if not response.status:
                raise Exception(f"Batch embedding failed: {response.error}")
            return [
                {
                    "dense": list(embedding.dense_values) if return_dense else None,
                    "sparse": dict(embedding.sparse_values) if return_sparse else None
                }
                for embedding in response.embeddings
            ]

    async def get_rerank_scores(self, query: str, corpus: List[str], 
                                target_topic: str, normalize: bool = True) -> List[float]:
        async with self.quick_embed_client.create_grpc_stub() as stub:
            request = TextRerankScoresRequest(
                query=query,
                target_topic=target_topic,
                corpus=corpus,
                normalize=normalize
            )
            response: TextRerankScoresResponse = await stub.getTextRerankScores(request)
            if not response.status:
                raise Exception(f"Reranking failed: {response.error}")
            return list(response.scores)

    