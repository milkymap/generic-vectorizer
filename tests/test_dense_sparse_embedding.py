import pytest
from unittest.mock import Mock, patch
import numpy as np
from google.protobuf.message import Message
from generic_vectorizer.strategies.embedding.bge_m3 import BGEM3FlagModelStrategy
from generic_vectorizer.grpc_server.interfaces.strategies_pb2 import TextEmbeddingRequest, TextEmbeddingResponse, TextBatchEmbeddingRequest, TextBatchEmbeddingResponse, Embedding

@pytest.fixture
def mock_bge_m3_flag_model():
    with patch('generic_vectorizer.strategies.embedding.bge_m3.BGEM3FlagModel') as mock:
        yield mock

def test_embedding(mock_bge_m3_flag_model):
    options = {
        "model_name_or_path": "BAAI/bge-m3",
        "device": "cpu",
        "use_fp16": False,
        "pooling_method": "cls"
    }
    strategy = BGEM3FlagModelStrategy(options)
    
    # Mock the encode method
    mock_embedding = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    strategy.model.encode.return_value = {
        'dense_vecs': mock_embedding
    }
    
    # Create a sample request
    request = TextEmbeddingRequest(
        text="Sample text for embedding",
        chunk_size=512,
        return_dense=True,
        return_sparse=False
    )
    encoded_request = request.SerializeToString()

    # Process the request
    response = strategy.process(b"TEXT", encoded_request)
    # Verify the response
    assert isinstance(response, TextEmbeddingResponse)
    assert response.status == True
    assert response.error == ""
    assert len(response.embedding.dense_values) == 3
    assert response.embedding.dense_values == pytest.approx(mock_embedding[0], abs=1e-6)
    assert not response.embedding.sparse_values

def test_sparse(mock_bge_m3_flag_model):
    options = {
        "model_name_or_path": "BAAI/bge-m3",
        "device": "cpu",
        "use_fp16": False,
        "pooling_method": "cls"
    }
    strategy = BGEM3FlagModelStrategy(options)
    
    # Mock the encode method
    strategy.model.encode.return_value = {
        'lexical_weights': [{'word1': 0.5, 'word2': 0.3, 'word3': 0.6}]
    }
    
    # Create a sample request
    request = TextEmbeddingRequest(
        text="Sample text for sparse embedding",
        chunk_size=512,
        return_dense=False,
        return_sparse=True
    )
    encoded_request = request.SerializeToString()

    # Process the request
    response = strategy.process(b"TEXT", encoded_request)

    # Verify the response
    assert isinstance(response, TextEmbeddingResponse)
    assert response.status == True
    assert response.error == ""
    assert not response.embedding.dense_values
    expected_sparse = {'word1': 0.5, 'word2': 0.3, 'word3': 0.6}
    assert response.embedding.sparse_values.keys() == expected_sparse.keys()
    for key in expected_sparse:
        assert response.embedding.sparse_values[key] == pytest.approx(expected_sparse[key], abs=1e-6)

def test_batch_embedding(mock_bge_m3_flag_model):
    options = {
        "model_name_or_path": "BAAI/bge-m3",
        "device": "cpu",
        "use_fp16": False,
        "pooling_method": "cls"
    }
    strategy = BGEM3FlagModelStrategy(options)
    
    # Mock the encode method
    mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    strategy.model.encode.return_value = {
        'dense_vecs': mock_embeddings
    }
    
    # Create a sample batch request
    request = TextBatchEmbeddingRequest(
        texts=["Sample text 1", "Sample text 2"],
        chunk_size=512,
        return_dense=True,
        return_sparse=False
    )
    encoded_request = request.SerializeToString()

    # Process the request
    response = strategy.process(b"TEXT_BATCH", encoded_request)

    # Verify the response
    assert isinstance(response, TextBatchEmbeddingResponse)
    assert response.status == True
    assert response.error == ""
    assert len(response.embeddings) == 2
    assert response.embeddings[0].dense_values == pytest.approx(mock_embeddings[0], abs=1e-6)
    assert response.embeddings[1].dense_values == pytest.approx(mock_embeddings[1], abs=1e-6)
    assert not response.embeddings[0].sparse_values
    assert not response.embeddings[1].sparse_values