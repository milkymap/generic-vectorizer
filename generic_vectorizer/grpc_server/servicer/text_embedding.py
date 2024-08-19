

import asyncio 

import zmq 
import zmq.asyncio as aiozmq 

from grpc import ServicerContext, StatusCode
from typing import List, Tuple, Dict, Optional, AsyncGenerator

from generic_vectorizer.grpc_server.interfaces import strategies_pb2, strategies_pb2_grpc

from generic_vectorizer.log import logger 
from contextlib import asynccontextmanager

class TextEmbeddingServicer(strategies_pb2_grpc.TextEmbeddingServicer):
    def __init__(self, ctx:aiozmq.Context, client2broker_addr:str, shared_semaphore:asyncio.Semaphore):
        self.ctx = ctx  
        self.cleint2broker_addr = client2broker_addr
        self.shared_semaphore = shared_semaphore
    
    async def _wait_socket_response_in_loop(self, dealer_socket:aiozmq.Socket) -> bytes:
        while True:
            try:
                incoming_signal = await dealer_socket.poll(timeout=100)
                if incoming_signal != zmq.POLLIN:
                    continue
                _, encoded_incoming_res = await dealer_socket.recv_multipart()
                return encoded_incoming_res 
            except asyncio.CancelledError:
                break 
        
    @asynccontextmanager
    async def _create_socket(self, addr:str) -> AsyncGenerator[aiozmq.Socket, None]:
        dealer_socket:aiozmq.Socket = self.ctx.socket(socket_type=zmq.DEALER)
        caught_exc:Optional[Exception] = None 
        try:
            dealer_socket.connect(addr=addr)
            yield dealer_socket
        except Exception as e:
            caught_exc = e
            logger.error(e)
        finally:
            dealer_socket.close(linger=0)

        if caught_exc is not None:
            raise caught_exc
    
    async def getTextRerankScores(self, request:strategies_pb2.TextRerankScoresRequest, context:ServicerContext):
        async with self.shared_semaphore:
            try:
                async with self._create_socket(addr=self.cleint2broker_addr) as dealer_socket:
                    encoded_req = request.SerializeToString()
                    await dealer_socket.send_multipart([b'', request.target_topic.encode(), b'', encoded_req])
                    encoded_res = await self._wait_socket_response_in_loop(dealer_socket=dealer_socket)
            except Exception as e:
                logger.warning(e)
                await context.abort(code=StatusCode.INTERNAL, details=str(e))
            
            if encoded_res.startswith(b'INTERNAL-ERROR:'):
                return strategies_pb2.TextRerankScoresResponse(
                    status=False,
                    error=encoded_res.decode()
                )

            plain_res = strategies_pb2.TextRerankScoresResponse()
            plain_res.ParseFromString(encoded_res)

            return plain_res
        
    async def getTextEmbedding(self, request:strategies_pb2.TextEmbeddingRequest, context:ServicerContext):
        async with self.shared_semaphore:
            try:
                async with self._create_socket(addr=self.cleint2broker_addr) as dealer_socket:
                    encoded_req = request.SerializeToString()
                    await dealer_socket.send_multipart([b'', request.target_topic.encode(), b'TEXT', encoded_req])
                    encoded_res = await self._wait_socket_response_in_loop(dealer_socket=dealer_socket)
            except Exception as e:
                logger.warning(e)
                await context.abort(code=StatusCode.INTERNAL, details=str(e))
            
            if encoded_res.startswith(b'INTERNAL-ERROR:'):
                return strategies_pb2.TextEmbeddingResponse(
                    status=False,
                    error=encoded_res.decode()
                )

            plain_res = strategies_pb2.TextEmbeddingResponse()
            plain_res.ParseFromString(encoded_res)

            return plain_res
        
    async def getTextBatchEmbedding(self, request:strategies_pb2.TextBatchEmbeddingRequest, context:ServicerContext):
        async with self.shared_semaphore:
            try:
                async with self._create_socket(addr=self.cleint2broker_addr) as dealer_socket:
                    encoded_req = request.SerializeToString()
                    await dealer_socket.send_multipart([b'', request.target_topic.encode(), b'TEXT_BATCH', encoded_req])
                    encoded_res = await self._wait_socket_response_in_loop(dealer_socket=dealer_socket)
            except Exception as e:
                logger.warning(e)
                await context.abort(code=StatusCode.INTERNAL, details=str(e))
            
            if encoded_res.startswith(b'INTERNAL-ERROR:'):
                return strategies_pb2.TextEmbeddingResponse(
                    status=False,
                    error=encoded_res.decode()
                )

            plain_res = strategies_pb2.TextBatchEmbeddingResponse()
            plain_res.ParseFromString(encoded_res)

            return plain_res
    

   