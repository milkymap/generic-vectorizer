from typing import AsyncGenerator

import grpc 

from operator import attrgetter
from typing import Optional, List, Union
from embedding_server.grpc_server.interfaces import strategies_pb2_grpc
from embedding_server.log import logger 

from contextlib import asynccontextmanager

class AsyncQuickEmbedClient:
    def __init__(self, grpc_server_address:str):
        self.grpc_server_address = grpc_server_address 

    @asynccontextmanager
    async def create_grpc_stub(self, stub_type:str='TextEmbeddingStub') -> AsyncGenerator[strategies_pb2_grpc.TextEmbeddingStub, None]: 
        async with grpc.aio.insecure_channel(target=self.grpc_server_address) as channel:
            try:
                stub = attrgetter(stub_type)(strategies_pb2_grpc)(channel=channel)
                yield stub 
            except Exception as e:
                logger.error(e)
                raise 
        # close the stub 
    
