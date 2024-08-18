from google.protobuf.message import Message
from ..abstract_strategy import ABCStrategy

from FlagEmbedding import FlagReranker
from embedding_server.model_schema import FlagRerankerConfig

from embedding_server.grpc_server.interfaces.strategies_pb2 import TextRerankScoresRequest, TextRerankScoresResponse

from typing import Dict, Any 
from embedding_server.log import logger 

from itertools import zip_longest

class FlagRerankerStrategy(ABCStrategy):
    def __init__(self, options:Dict[str, Any]) -> None:
        config = FlagRerankerConfig(**options)
        self.model = FlagReranker(**config.model_dump())
        
    def process(self, task_type: bytes, encoded_message: bytes) -> Message:
        try:
            plain_message = TextRerankScoresRequest()
            plain_message.ParseFromString(encoded_message)
            sentence_pairs = list(zip_longest([plain_message.query], plain_message.corpus, fillvalue=plain_message.query))
            scores = self.model.compute_score(sentence_pairs=sentence_pairs, normalize=plain_message.normalize)
            response = TextRerankScoresResponse(status=True, error=None, scores=scores)
        except Exception as e:
            response = TextRerankScoresResponse(status=False, error=str(e), scores=None)
            logger.error(e) 
        
        return response