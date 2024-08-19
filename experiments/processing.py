import click 
import asyncio

from os import path 
from glob import glob 

from typing import List, Dict 
from generic_vectorizer.log import logger 
from generic_vectorizer.client import AsyncEmbeddingClient

from tqdm import tqdm

async def batch_processing(path2corpus:str, grpc_server_address:str, batch_size:int, target_topic:str):
    file_paths:List[str] = sorted(glob(path.join(path2corpus, '*.txt')))
    client = AsyncEmbeddingClient(grpc_server_address=grpc_server_address)
    logger.info(f'{len(file_paths)} files were found')

    accumulator:List[str] = []
    for counter in tqdm(range(0, len(file_paths), batch_size)):
        for index in range(counter, counter + batch_size):
            path2file = file_paths[index]
            with open(file=path2file, mode='r') as file_pointer:
                content = file_pointer.read()
                accumulator.append(content)
        
        batch_embedding_response = await client.get_batch_embedding(
            texts=accumulator,
            target_topic=target_topic,
            return_dense=True,
            return_sparse=True
        )

@click.command()
@click.option('--path2corpus', type=click.Path(exists=True, file_okay=False), required=True)
@click.option('--grpc_server_address', default='localhost:5000')
@click.option('--batch_size', default=8, help='size of the batch')
@click.option('--target_topic', required=True, help='embedding service topic')
def process_files(path2corpus:str, grpc_server_address:str, batch_size:int, target_topic:str):
    asyncio.run(batch_processing(path2corpus, grpc_server_address, batch_size, target_topic))


if __name__ == '__main__':
    process_files()