import pytest
from unittest.mock import Mock, patch
from google.protobuf.message import Message
from generic_vectorizer.strategies.reranker.flag_reranker import FlagRerankerStrategy
from generic_vectorizer.grpc_server.interfaces.strategies_pb2 import TextRerankScoresRequest, TextRerankScoresResponse
from generic_vectorizer.typing import FlagRerankerConfig

@pytest.fixture
def mock_flag_reranker():
    with patch('generic_vectorizer.strategies.reranker.flag_reranker.FlagReranker') as mock:
        yield mock

def test_flag_reranker_strategy_initialization(mock_flag_reranker):
    options = {
        "model_name_or_path": "BAAI/bge-reranker-v2-m3",
        "device": "cpu",
        "use_fp16": False
    }
    strategy = FlagRerankerStrategy(options)
    
    mock_flag_reranker.assert_called_once_with(**FlagRerankerConfig(**options).model_dump())
    assert isinstance(strategy, FlagRerankerStrategy)

def test_flag_reranker_strategy_process_success(mock_flag_reranker):
    options = {
        "model_name_or_path": "BAAI/bge-reranker-v2-m3",
        "device": "cpu",
        "use_fp16": False
    }
    strategy = FlagRerankerStrategy(options)
    
    # Mock the compute_score method
    strategy.model.compute_score.return_value = [0.5, 0.7, 0.9]

    # Create a sample request
    request = TextRerankScoresRequest(
        query="Sample query",
        corpus=["Document 1", "Document 2", "Document 3"],
        normalize=True
    )
    encoded_request = request.SerializeToString()

    # Process the request
    response = strategy.process(b"rerank", encoded_request)

    # Verify the response
    assert isinstance(response, TextRerankScoresResponse)
    assert response.status == True
    assert response.error is ""
    assert response.scores == pytest.approx([0.5, 0.7, 0.9], abs=1e-6)

    # Verify that compute_score was called correctly
    strategy.model.compute_score.assert_called_once_with(
        sentence_pairs=[
            ("Sample query", "Document 1"),
            ("Sample query", "Document 2"),
            ("Sample query", "Document 3")
        ],
        normalize=True
    )

def test_flag_reranker_strategy_process_failure(mock_flag_reranker):
    options = {
        "model_name_or_path": "BAAI/bge-reranker-v2-m3",
        "device": "cpu",
        "use_fp16": False
    }
    strategy = FlagRerankerStrategy(options)
    
    # Mock the compute_score method to raise an exception
    strategy.model.compute_score.side_effect = Exception("Test error")

    # Create a sample request
    request = TextRerankScoresRequest(
        query="Sample query",
        corpus=["Document 1", "Document 2", "Document 3"],
        normalize=True
    )
    encoded_request = request.SerializeToString()

    # Process the request
    response = strategy.process(b"rerank", encoded_request)

    # Verify the response
    assert isinstance(response, TextRerankScoresResponse)
    assert response.status == False
    assert response.error == "Test error"
    assert len(response.scores) == 0


@pytest.mark.parametrize("query,corpus,normalize,expected_pairs", [
    ("Query", ["Doc1", "Doc2"], True, [("Query", "Doc1"), ("Query", "Doc2")]),
    ("Query", ["Doc1"], False, [("Query", "Doc1")]),
    ("Query", [], True, [("Query", "Query")]),  # Change this line
])
def test_flag_reranker_strategy_sentence_pairs(mock_flag_reranker, query, corpus, normalize, expected_pairs):
    options = {
        "model_name_or_path": "BAAI/bge-reranker-v2-m3",
        "device": "cpu",
        "use_fp16": False
    }
    strategy = FlagRerankerStrategy(options)
    
    # Mock the compute_score method
    strategy.model.compute_score.return_value = [0.5] * len(expected_pairs)

    # Create a sample request
    request = TextRerankScoresRequest(query=query, corpus=corpus, normalize=normalize)
    encoded_request = request.SerializeToString()

    # Process the request
    strategy.process(b"rerank", encoded_request)

    # Verify that compute_score was called with the correct sentence pairs
    strategy.model.compute_score.assert_called_once_with(
        sentence_pairs=expected_pairs,
        normalize=normalize
    )