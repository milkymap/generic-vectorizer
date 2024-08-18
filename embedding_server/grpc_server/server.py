import asyncio 
import grpc 

import zmq 
import zmq.asyncio as aiozmq 

from hashlib import sha256

from embedding_server.grpc_server.interfaces import strategies_pb2_grpc
from embedding_server.grpc_server.servicer.text_embedding import TextEmbeddingServicer

from embedding_server.log import logger 

from embedding_server.model_schema import EmbedderModelConfig

from typing import List, Dict, Tuple 
from typing_extensions import Self 

from time import time 

class GRPCServer:
    _CLIENT2BORKER_ADDR:str='inproc://client2broker'
    _BROKER2ROUTER_ADDR:str='inproc://broker2router'

    def __init__(self, max_concurrent_requests:int=512, request_timeout:int=30):
        self.max_concurrent_requests = max_concurrent_requests 
        self.request_timeout = request_timeout
        self.topic2queue_hmap:Dict[str, asyncio.Queue] = {}

    async def __aenter__(self) -> Self:
        self.ctx = aiozmq.Context()
        self.ctx.set(zmq.MAX_SOCKETS, self.max_concurrent_requests)
        self.shared_semaphore = asyncio.Semaphore(int(0.7 * self.max_concurrent_requests))
        return self 
    
    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None:
            logger.error(exc_value)
            logger.exception(traceback)
        self.ctx.term()
    
    async def listen(self, embedder_model_configs:List[EmbedderModelConfig], grpc_server_address:str, grace:int=5) -> None:  
        server = grpc.aio.server()
        text_embedding_servicer = TextEmbeddingServicer(ctx=self.ctx, client2broker_addr=GRPCServer._CLIENT2BORKER_ADDR, shared_semaphore=self.shared_semaphore)  #inproc 
        strategies_pb2_grpc.add_TextEmbeddingServicer_to_server(
            servicer=text_embedding_servicer,
            server=server
        )
        server.add_insecure_port(address=grpc_server_address)
        await server.start()
        
        exchange_task = asyncio.create_task(
            self.broker(GRPCServer._CLIENT2BORKER_ADDR, GRPCServer._BROKER2ROUTER_ADDR)
        ) 
        
        proxies:List[asyncio.Task] = []
        for cfg in embedder_model_configs:
            self.topic2queue_hmap[cfg.target_topic] = asyncio.Queue()
            router2worker_addr = cfg.zmq_tcp_address or f'ipc:///tmp/router2worker_{cfg.target_topic}.ipc'  
            proxies.append(
                asyncio.create_task(
                    self.router(cfg.target_topic, GRPCServer._BROKER2ROUTER_ADDR, router2worker_addr)
                )
            )
        
        await exchange_task

        try:
            _ = await asyncio.gather(*proxies, return_exceptions=True)
        except asyncio.CancelledError:
            pass 
        
        try:
            await server.stop(grace=grace)
        except asyncio.CancelledError:
            pass 
    
    async def broker(self, client2broker_addr:str, broker2router_addr:str):
        client2broker_router_socket:aiozmq.Socket = self.ctx.socket(socket_type=zmq.ROUTER)
        broker2router_puller_socket:aiozmq.Socket = self.ctx.socket(socket_type=zmq.PULL)

        client2broker_router_socket.bind(addr=client2broker_addr)
        broker2router_puller_socket.bind(addr=broker2router_addr)

        poller = aiozmq.Poller()
        poller.register(client2broker_router_socket, zmq.POLLIN)
        poller.register(broker2router_puller_socket, zmq.POLLIN)

        while True:
            try:
                socket_hmap:Dict[zmq.Socket, int] = dict(await poller.poll(timeout=1000))    
                if socket_hmap.get(client2broker_router_socket, None) == zmq.POLLIN:
                    incoming_req:Tuple[bytes, bytes, bytes, bytes, bytes] = await client2broker_router_socket.recv_multipart()
                    source_client_id, _, encoded_topic, encoded_task_type, encoded_client_message = incoming_req
                    target_queue = self.topic2queue_hmap.get(encoded_topic.decode(), None)
                    if target_queue is None:
                        await client2broker_router_socket.send_multipart([
                            source_client_id, b'', 'INTERNAL-ERROR:{} is not a valid topic'.format(encoded_topic.decode()).encode()
                        ])
                    else:
                        await target_queue.put((source_client_id, encoded_task_type, encoded_client_message))

                if socket_hmap.get(broker2router_puller_socket, None) == zmq.POLLIN:
                    target_client_id, encoded_worker_message = await broker2router_puller_socket.recv_pyobj()
                    await client2broker_router_socket.send_multipart([target_client_id, b'', encoded_worker_message])

            except asyncio.CancelledError:
                break 
            except Exception as e:
                logger.warning(e)
                break 
        

        client2broker_router_socket.close(linger=0)
        broker2router_puller_socket.close(linger=0)
 
    
    async def router(self, topic:str, broker2router_addr:str, router2worker_addr:str):
        broker2router_pusher_socket:aiozmq.Socket = self.ctx.socket(socket_type=zmq.PUSH)
        router2worker_router_socket:aiozmq.Socket = self.ctx.socket(socket_type=zmq.ROUTER)

        broker2router_pusher_socket.connect(addr=broker2router_addr)
        router2worker_router_socket.bind(addr=router2worker_addr)

        poller = aiozmq.Poller()
        poller.register(router2worker_router_socket, zmq.POLLIN)

        worker_ids:List[bytes] = []
        marker = time()
        while True:
            try:
                socket_hmap:Dict[zmq.Socket, int] = dict(await poller.poll(timeout=1000))
                duration = time() - marker 
                if duration > 5:
                    logger.info(f'grpc server topic : {topic} is running with {len(worker_ids)} background transformer workers')
                    marker = time()

                target_queue = self.topic2queue_hmap.get(topic, None)
                assert target_queue is not None, f'{topic} must have a queue...!'    
                if not target_queue.empty() and len(worker_ids) > 0:
                    target_worker_id = worker_ids.pop(0)
                    source_client_id, encoded_task_type, encoded_client_message = await target_queue.get()
                    await router2worker_router_socket.send_multipart(
                     [target_worker_id, b'', source_client_id, encoded_task_type, encoded_client_message]
                 )
                    
                if socket_hmap.get(router2worker_router_socket, None) == zmq.POLLIN:
                    incoming_res = await router2worker_router_socket.recv_multipart()
                    source_worker_id, _, encoded_worker_signal, target_client_id, encoded_worker_message = incoming_res
                    
                    if encoded_worker_signal == b'HANDSHAKE':
                        worker_ids.append(source_worker_id) 
                        continue

                    if encoded_worker_signal == b'RESPONSE':
                        await broker2router_pusher_socket.send_pyobj((target_client_id, encoded_worker_message))
                         
            except asyncio.CancelledError:
                logger.info(f'grpc server topic : {topic} cancelled')
                break 
            except Exception as e:
                logger.error(e)
                break 
        
        broker2router_pusher_socket.close(linger=0)
        router2worker_router_socket.close(linger=0)


def run_grpc_server(grpc_server_address:str, embedder_model_configs:List[EmbedderModelConfig], grace:int=5):
    async def main():
        async with GRPCServer() as server:
            await server.listen(
                embedder_model_configs=embedder_model_configs,
                grpc_server_address=grpc_server_address,
                grace=grace
            )
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass 