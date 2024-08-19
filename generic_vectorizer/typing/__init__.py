from enum import Enum 
from typing import Optional, List, Any, Dict 

from pydantic import BaseModel

from .reranker import FlagRerankerConfig
from .bge_embedding import BGEM3FlagModelConfig

class EmbedderModelType(str, Enum):
    BGE_RERANKER_MODEL:str='FlagRerankerStrategy'
    BGE_M3_EMBEDDING_MODEL:str='BGEM3FlagModelStrategy'
    
class EmbedderModelConfig(BaseModel):
    embedder_model_type:EmbedderModelType
    target_topic:str
    nb_instances:int=1
    zmq_tcp_address:Optional[str]=None
    options:Dict[str, Any] 