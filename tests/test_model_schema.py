import pytest
from pydantic import ValidationError
from generic_vectorizer.typing import (
    FlagRerankerConfig, 
    BGEM3FlagModelConfig, 
    EmbedderModelConfig, 
    EmbedderModelType
)

# Tests for FlagRerankerConfig
def test_flag_reranker_config_valid():
    config = FlagRerankerConfig(
        model_name_or_path='BAAI/bge-reranker-v2-m3',
        device='cpu',
        use_fp16=True,
        cache_dir=None
    )
    assert config.model_name_or_path == 'BAAI/bge-reranker-v2-m3'
    assert config.device == 'cpu'
    assert config.use_fp16 == True
    assert config.cache_dir == None

def test_flag_reranker_config_invalid_model():
    with pytest.raises(ValidationError):
        FlagRerankerConfig(model_name_or_path='invalid_model')

# Tests for BGEM3FlagModelConfig
def test_bgem3_flag_model_config_valid():
    config = BGEM3FlagModelConfig(
        model_name_or_path='BAAI/bge-m3',
        device='cpu',
        use_fp16=True,
        pooling_method='cls'
    )
    assert config.model_name_or_path == 'BAAI/bge-m3'
    assert config.device == 'cpu'
    assert config.use_fp16 == True
    assert config.pooling_method == 'cls'

def test_bgem3_flag_model_config_invalid_model():
    with pytest.raises(ValidationError):
        BGEM3FlagModelConfig(model_name_or_path='invalid_model')

def test_bgem3_flag_model_config_invalid_pooling_method():
    with pytest.raises(ValidationError):
        BGEM3FlagModelConfig(pooling_method='invalid_method')

# Tests for EmbedderModelConfig
def test_embedder_model_config_valid():
    config = EmbedderModelConfig(
        embedder_model_type=EmbedderModelType.BGE_RERANKER_MODEL,
        target_topic='test_topic',
        nb_instances=2,
        zmq_tcp_address='tcp://localhost:5555',
        options={'key': 'value'}
    )
    assert config.embedder_model_type == EmbedderModelType.BGE_RERANKER_MODEL
    assert config.target_topic == 'test_topic'
    assert config.nb_instances == 2
    assert config.zmq_tcp_address == 'tcp://localhost:5555'
    assert config.options == {'key': 'value'}

def test_embedder_model_config_minimal():
    config = EmbedderModelConfig(
        embedder_model_type=EmbedderModelType.BGE_M3_EMBEDDING_MODEL,
        target_topic='test_topic',
        options={}
    )
    assert config.embedder_model_type == EmbedderModelType.BGE_M3_EMBEDDING_MODEL
    assert config.target_topic == 'test_topic'
    assert config.nb_instances == 1
    assert config.zmq_tcp_address == None
    assert config.options == {}

def test_embedder_model_config_invalid_type():
    with pytest.raises(ValidationError):
        EmbedderModelConfig(
            embedder_model_type='InvalidType',
            target_topic='test_topic',
            options={}
        )