import click 
import asyncio
import json
from typing import List, Dict

from .vectorizer import Vectorizer
from .log import logger 
from .typing import EmbedderModelConfig, EmbedderModelType

@click.group(chain=True, invoke_without_command=True)
@click.pass_context
def handler(ctx: click.core.Context):
    ctx.ensure_object(dict)

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def create_embedder_model_configs(config: Dict) -> List[EmbedderModelConfig]:
    return [
        EmbedderModelConfig(
            embedder_model_type=EmbedderModelType[model_config['embedder_model_type']],
            target_topic=model_config['target_topic'],
            nb_instances=model_config['nb_instances'],
            options=model_config['options'],
            zmq_tcp_address=model_config.get('zmq_tcp_address')
        )
        for model_config in config['embedder_model_configs']
    ]

@handler.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to the configuration JSON file')
@click.pass_context
def launch_engine(ctx: click.core.Context, config: str):
    try:
        config_data = load_config(config)
        embedder_model_configs = create_embedder_model_configs(config_data)
        
        vectorizer = Vectorizer(
            grpc_server_address=config_data['grpc_server_address'],
            embedder_model_configs=embedder_model_configs,
            max_concurrent_requests=config_data['max_concurrent_requests'],
            request_timeout=config_data['request_timeout']
        )
        vectorizer.listen()
    except Exception as e:
        logger.error(f"Error launching engine: {str(e)}")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    handler()