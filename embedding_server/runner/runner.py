
import multiprocessing as mp 
from embedding_server.grpc_server.server import run_grpc_server
from embedding_server.background_workers.embedder import EmbedderPool

from embedding_server.model_schema import EmbedderModelConfig, EmbedderModelType

from typing import List

import re 
import socket 

from collections import Counter

class Runner:
    def __init__(self, grpc_server_address:str, embedder_model_configs:List[EmbedderModelConfig]):
        self.grpc_server_address = grpc_server_address
        self.validate_topics(embedder_model_configs)
        self.validate_zmq_tcp_addresses(embedder_model_configs)
        self.embedder_model_configs = embedder_model_configs 
    
    def validate_topics(self, embedder_model_configs: List[EmbedderModelConfig]) -> None:
        topics = [cfg.target_topic for cfg in embedder_model_configs]
        self.check_duplicates(topics)

    def check_duplicates(self, topics: List[str]) -> None:
        topic_counts = Counter(topics)
        duplicates = {topic: count for topic, count in topic_counts.items() if count > 1}
        
        if duplicates:
            duplicate_info = ', '.join(f"'{topic}' (occurs {count} times)" for topic, count in duplicates.items())
            raise ValueError(f"Duplication is not allowed for topics. Duplicates found: {duplicate_info}")



    def is_port_available(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return True
            except socket.error:
                return False

    def validate_zmq_tcp_addresses(self, embedder_model_configs: List[EmbedderModelConfig]) -> List[str]:
        zmq_tcp_addresses = [cfg.zmq_tcp_address for cfg in embedder_model_configs if cfg.zmq_tcp_address is not None]
        if not zmq_tcp_addresses:
            return []

        valid_addresses = []
        for address in zmq_tcp_addresses:
            if not re.match(r'^tcp://\*:\d+$', address):
                raise ValueError(f"Invalid ZeroMQ TCP address format: {address}. Must be in the format 'tcp://*:port_number'")
            
            port = int(address.split(':')[-1])
            if not self.is_port_available(port):
                raise ValueError(f"Port {port} is not available for address: {address}")
            valid_addresses.append(address)
        return valid_addresses

    def listen(self):
        grpc_process = mp.Process(
            target=run_grpc_server, 
            args=[self.grpc_server_address, self.embedder_model_configs]
        )
        grpc_process.start()
        embedder_pool = EmbedderPool(embedder_model_configs=self.embedder_model_configs)
        embedder_pool.launch_workers()

        try:
            grpc_process.join()
        except KeyboardInterrupt:
            grpc_process.terminate()
            grpc_process.join()
