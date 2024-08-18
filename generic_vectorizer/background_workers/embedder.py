import zmq 
import multiprocessing as mp
from operator import itemgetter, attrgetter

import signal 
from generic_vectorizer.log import logger 

from time import sleep 

from typing import List, Dict, Type, Any  

from ..strategies.abstract_strategy import ABCStrategy
from ..typing import EmbedderModelConfig

import generic_vectorizer.strategies as stratref

class EmbedderPool:
    def __init__(self, embedder_model_configs:List[EmbedderModelConfig]) -> None:
        assert len(embedder_model_configs) > 0, f"embedder_model_configs length must be grater then 0"
        self.embedder_model_configs = embedder_model_configs

        for config in embedder_model_configs:
            try:
                attrgetter(config.embedder_model_type)(stratref)
            except Exception as e:
                logger.error(e)
                exit(0)
    
    def build_strategy(self, strategy:Type[ABCStrategy], options:Dict[str, Any]) -> ABCStrategy:
        action = strategy(options)
        return action
    
    def processing(self, worker_id:str, config:EmbedderModelConfig):
        signal.signal(
            signalnum=signal.SIGTERM,
            handler=lambda signal_n, frame: signal.raise_signal(signal.SIGINT)
        ) 

        try:
            strategy = attrgetter(config.embedder_model_type)(stratref)
            action = self.build_strategy(strategy, config.options)
        except Exception as e:
            logger.error(e)
            exit(-1) 

        logger.info(f'{worker_id} : model was loaded | options >> {config.options}')
        socket_connected:bool = False 
        try:
            ctx = zmq.Context()
            dealer_socket:zmq.Socket = ctx.socket(socket_type=zmq.DEALER)
            logger.info(f'{worker_id} socket was created')
            topic = config.target_topic
            broker2worker_addr = f'ipc:///tmp/router2worker_{topic}.ipc' 
            if config.zmq_tcp_address is not None:
                broker2worker_addr = config.zmq_tcp_address.replace('*', 'localhost')
            dealer_socket.connect(addr=broker2worker_addr)  # add topics based on the flag_model_type 
            socket_connected = True 
            logger.info(f'{worker_id} has performed the handshake with backend router')
        except Exception as e:
            logger.error(e)
            if socket_connected: 
                dealer_socket.close(linger=0)
            exit(-1)

        dealer_socket.send_multipart([b'', b'HANDSHAKE', b'', b''])
        logger.info(f'{worker_id} is running')
        
        while True:
            try:
                incoming_signal = dealer_socket.poll(timeout=5000)
                if incoming_signal != zmq.POLLIN:
                    continue

                _, target_client_socket_id, encoded_task_type, encoded_client_message = dealer_socket.recv_multipart()
                
                try:
                    plain_worker_message = action.process(encoded_task_type, encoded_client_message)
                    encoded_worker_message = plain_worker_message.SerializeToString()  # add exception around action.process
                    dealer_socket.send_multipart([b'', b'RESPONSE', target_client_socket_id], flags=zmq.SNDMORE)
                    dealer_socket.send(encoded_worker_message)
                except Exception as e:
                    logger.warning(e)
                    dealer_socket.send_multipart([b'', b'RESPONSE', target_client_socket_id], flags=zmq.SNDMORE)
                    dealer_socket.send_string(f'INTERNAL-ERROR:{str(e)}')  # USE ERROR TOPIC INSTEAD OF RESPONSE
                logger.info(f'{worker_id} has consumed a message')   
                dealer_socket.send_multipart([b'', b'HANDSHAKE', b'', b''])
            except KeyboardInterrupt:
                break 
            except Exception as e:
                logger.error(e)
                break 
        
        dealer_socket.close(linger=0)
        ctx.term()
        logger.info(f'{worker_id} model shutdown')

    def launch_workers(self) -> None:
        processes:List[mp.Process] = []
        worker_id = 0
        for config in self.embedder_model_configs:
            for _ in range(config.nb_instances):
                process_ = mp.Process(target=self.processing, args=[f'worker-{worker_id:03d}', config])
                processes.append(process_)
                processes[-1].start()
                worker_id += 1 

        while True:
            try:
                exitcodes = [ process_.exitcode is not None for process_ in processes ]
                if any(exitcodes):
                    break 
                sleep(1)  # wait 1 second to keep the loop
            except KeyboardInterrupt:
                for process_ in processes:
                    process_.join()  # wait for worker to handle the SIGINT
                break 
            except Exception as e:
                logger.error(e)
                break 
        
        for process_ in processes:
            if process_.is_alive():
                process_.terminate()  # send SIGTERM => SIGINT 
                process_.join()  # wait for worker to handle the signal 