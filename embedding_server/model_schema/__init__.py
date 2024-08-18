from .bge_embedding import BGEM3FlagModelConfig
from .reranker import FlagRerankerConfig

from enum import Enum 
from typing import Optional, List, Any, Dict 

from pydantic import BaseModel

class EmbedderModelType(str, Enum):
    BGE_RERANKER_MODEL:str='FlagRerankerStrategy'
    BGE_M3_EMBEDDING_MODEL:str='BGEM3FlagModelStrategy'
    SENTENCE_TRANSFORMERS_DENSE_MODEL:str='SBERTModelStrategy'
    CLIP_VIT_MODEL:str='ClipViTModelStrategy'

class EmbedderModelConfig(BaseModel):
    embedder_model_type:EmbedderModelType
    target_topic:str
    nb_instances:int=1
    zmq_tcp_address:Optional[str]=None
    options:Dict[str, Any] 