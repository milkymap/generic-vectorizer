from typing import AsyncGenerator

import grpc 

from operator import attrgetter
from typing import Optional, List, Union
from embedding_server.grpc_server.interfaces import strategies_pb2_grpc
from embedding_server.log import logger 

from contextlib import asynccontextmanager
from embedding_server.settings.grpc_server_settings import GRPCServerSettings

class AsyncQuickEmbedClient:
    def __init__(self, grpc_server_settings:GRPCServerSettings):
        self.grpc_server_settings = grpc_server_settings 

    @asynccontextmanager
    async def create_grpc_stub(self, stub_type:str='TextEmbeddingStub') -> AsyncGenerator[strategies_pb2_grpc.TextEmbeddingStub, None]: 
        async with grpc.aio.insecure_channel(target=self.grpc_server_settings.target) as channel:
            try:
                stub = attrgetter(stub_type)(strategies_pb2_grpc)(channel=channel)
                yield stub 
            except Exception as e:
                logger.error(e)
                raise 
        # ckise the stub 
    
