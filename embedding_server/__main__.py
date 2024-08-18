import click 
import asyncio

from .settings.grpc_server_settings import GRPCServerSettings
from .runner import Runner
from .log import logger 
from .model_schema import EmbedderModelConfig, EmbedderModelType
from .client import AsyncQuickEmbedClient, TextEmbeddingRequest, TextEmbeddingResponse, TextRerankScoresRequest, TextRerankScoresResponse, TextBatchEmbeddingRequest, TextBatchEmbeddingResponse

from tqdm import tqdm
from typing import List 

from time import perf_counter

@click.group(chain=True, invoke_without_command=True)
@click.pass_context
def handler(ctx:click.core.Context):
    ctx.ensure_object(dict)
    ctx.obj['grpc_server_settings'] = GRPCServerSettings() 


@handler.command()
@click.pass_context
def launch_client(ctx:click.core.Context):
    grpc_server_settings = ctx.obj['grpc_server_settings']
    async def main():
        quick_embed_client = AsyncQuickEmbedClient(
            grpc_server_settings=grpc_server_settings
        )

        query = """Jean Castex, former French Prime Minister, is becoming head of RATP (Paris public transport). His main challenges:

        1. Staff motivation/retention: Struggling with recruitment and retention due to work conditions.
        2. Restoring transport services: Need to improve frequency and quality, linked to staffing.
        3. Managing open competition transition: Preparing for market opening, causing employee concerns.
        4. Financial challenges: RATP and funder face difficulties due to COVID-19 and rising costs.

        Castex's political experience seen as asset for addressing these issues, especially in securing funding and navigating Paris public transport complexities."""

        async with quick_embed_client.create_grpc_stub() as stub:
            coros:List[asyncio.Task] = []
            for _ in range(1):
                coros.append(
                    stub.getTextBatchEmbedding(
                        TextBatchEmbeddingRequest(
                            embed_strategy=EmbedderModelType.BGE_M3_EMBEDDING_MODEL,
                            target_topic='bge_m3',
                            texts=[query] * 10,
                            chunk_size=512,
                            return_dense=True,
                            return_sparse=True
                        )
                    )
                )
            
            marker = perf_counter()
            responses = await asyncio.gather(*coros, return_exceptions=True)
            duration = perf_counter() - marker 
            print('nb res:', len(responses), 'duration:', duration, 'seconds')
            print(responses)

            response:TextRerankScoresResponse = await stub.getTextRerankScores(
                TextRerankScoresRequest(
                    normalize=True,
                    target_topic='bge_reranker',  # add instance validation grpc parse string
                    query='what is a panda?',
                    corpus=['HI!', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']
                )
            )

            print(response)
            
            
    asyncio.run(main=main())

@handler.command()
@click.pass_context
def launch_engine(ctx:click.core.Context):
    embedder_model_configs = [
        EmbedderModelConfig(
            embedder_model_type=EmbedderModelType.BGE_M3_EMBEDDING_MODEL,
            target_topic='bge_m3',
            nb_instances=3,
            options={
                'model_name_or_path': 'BAAI/bge-m3',
                'device': 'cuda:0'
            }
        ),
        EmbedderModelConfig(
            embedder_model_type=EmbedderModelType.BGE_RERANKER_MODEL,
            target_topic='bge_reranker',
            nb_instances=1,
            options={
                'model_name_or_path': 'BAAI/bge-reranker-v2-m3',
                'device': 'cpu'
            }
        )
    ]
    runner = Runner(
        grpc_server_address='[::]:1200',
        embedder_model_configs=embedder_model_configs
    )
    runner.listen()


if __name__ == '__main__':
    handler()