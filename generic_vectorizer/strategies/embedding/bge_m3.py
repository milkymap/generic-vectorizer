import numpy as np 

from typing import Any
from ..abstract_strategy import ABCStrategy

from functools import reduce 

from FlagEmbedding import BGEM3FlagModel
from generic_vectorizer.typing import BGEM3FlagModelConfig

from generic_vectorizer.grpc_server.interfaces.strategies_pb2 import TextEmbeddingRequest, TextEmbeddingResponse, Embedding
from generic_vectorizer.grpc_server.interfaces.strategies_pb2 import TextBatchEmbeddingRequest, TextBatchEmbeddingResponse

from typing import Dict, List, Tuple, Union, Callable 
from typing import Optional

from generic_vectorizer.log import logger 
from numpy.typing import NDArray

class BGEM3FlagModelStrategy(ABCStrategy):
    def __init__(self, options:Dict[str, Any]) -> None:
        config = BGEM3FlagModelConfig(**options)
        self.model = BGEM3FlagModel(**config.model_dump())
        self.tokenizer = self.model.tokenizer
        self.map_task2function:Dict[bytes, Callable[[bytes], Union[TextEmbeddingResponse, TextBatchEmbeddingRequest]]] = {
            b'TEXT': self._process_text,
            b'TEXT_BATCH': self._process_batch_texts
        }

    def to_chunks(self, text:str, chunk_size:int) -> List[str]:
        tokens = self.tokenizer.encode(text=text, add_special_tokens=False)
        accumulator:List[str] = []
        for counter in range(0, len(tokens), chunk_size):
            tokens_slice = tokens[counter:counter+chunk_size]
            text_chunk = self.tokenizer.decode(token_ids=tokens_slice) 
            accumulator.append(text_chunk)
        
        if len(accumulator) == 0:
            accumulator.append(' ')

        return accumulator

    def aggregate_embeddings(self, embeddings:NDArray) -> NDArray:
        if embeddings.shape[0] == 1:
            return embeddings[0]
        
        dot_scores = embeddings @ embeddings.T 
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        cosine_similarity_scores = dot_scores / (embedding_norms[None, :] * embedding_norms[:, None] + 1e-8)
        node_centrality_scores = np.sum(cosine_similarity_scores, axis=1, keepdims=True)
        text_embedding:NDArray = np.mean(node_centrality_scores * embeddings, axis=0)
        return text_embedding

    def _process_text(self, encoded_message:bytes) -> TextEmbeddingResponse:
        try:
            plain_message = TextEmbeddingRequest()
            plain_message.ParseFromString(encoded_message)
            assert plain_message.return_dense | plain_message.return_sparse == True, f'one of [return_dense or return_sparse] was not set!'
            
            sentences = self.to_chunks(text=plain_message.text, chunk_size=plain_message.chunk_size)
            embeddings_hmap:Dict = self.model.encode(sentences=sentences, return_dense=plain_message.return_dense, return_sparse=plain_message.return_sparse)
            dense_embeddings:Optional[NDArray] = embeddings_hmap.get('dense_vecs', None)
            
            if dense_embeddings is not None:
                dense_values = self.aggregate_embeddings(embeddings=dense_embeddings)
            else:
                dense_values = None 

            lexical_weights:Optional[List[Dict]] = embeddings_hmap.get('lexical_weights', None)
            sparse_values = None 
            if lexical_weights is not None:
                sparse_values = lexical_weights
                sparse_values = reduce(lambda acc, elm: {**acc, **elm}, sparse_values[1:], sparse_values[0])
                sparse_values = {key:val for key,val in sparse_values.items()}  # use max scores as best key in case of dupplication
            response = TextEmbeddingResponse(status=True, error=None, embedding=Embedding(dense_values=dense_values, sparse_values=sparse_values))        
        except Exception as e:
            logger.error(e)
            response = TextEmbeddingResponse(status=False, error=str(e), embedding=Embedding(dense_values=None, sparse_values=None))    

        return response

    def _process_batch_texts(self, encoded_message:bytes) -> TextBatchEmbeddingResponse:
        try:
            plain_message = TextBatchEmbeddingRequest()
            plain_message.ParseFromString(encoded_message) 
            assert plain_message.return_dense | plain_message.return_sparse == True, f'one of [return_dense or return_sparse] was not set!'
            accumulator:List[str] = []
            nb_chunks:List[int] = []
            for text in plain_message.texts:
                sentences = self.to_chunks(text=text, chunk_size=plain_message.chunk_size) 
                nb_chunks.append(len(sentences))
                accumulator.extend(sentences)
            
            embeddings_hmap = self.model.encode(sentences=accumulator, return_dense=plain_message.return_dense, return_sparse=plain_message.return_sparse)
            dense_embeddings:Optional[NDArray] = embeddings_hmap.get('dense_vecs', None)
            lexical_weights:Optional[List[Dict]] = embeddings_hmap.get('lexical_weights', None)
            text_embeddings_acc:List[Embedding] = []
                
            nb_embeddings = len(accumulator)
            counter = 0
            index = 0
            while counter < nb_embeddings:
                dense_values = None
                if dense_embeddings is not None:
                    embeddings_slice = dense_embeddings[counter:counter+nb_chunks[index], :]
                    dense_values = self.aggregate_embeddings(embeddings=embeddings_slice).tolist()
                
                sparse_values = None 
                if lexical_weights is not None:
                    sparse_values = lexical_weights[counter:counter+nb_chunks[index]]             
                    sparse_values = reduce(lambda acc, elm: {**acc, **elm}, sparse_values[1:], sparse_values[0])
                    sparse_values = {key:val for key,val in sparse_values.items()}

                text_embeddings_acc.append(Embedding(dense_values=dense_values, sparse_values=sparse_values))
                counter = counter + nb_chunks[index]
                index = index + 1
                
            response = TextBatchEmbeddingResponse(embeddings=text_embeddings_acc, status=True, error=None)
        except Exception as e:
            logger.error(e)
            response = TextEmbeddingResponse(status=False, error=str(e),embedding=Embedding(dense_values=None, sparse_values=None))    

        return response        

    def process(
            self, 
            task_type:bytes, 
            encoded_message: bytes
        ) -> Union[TextEmbeddingResponse, TextBatchEmbeddingResponse]:  
        assert task_type in [b'TEXT', b'TEXT_BATCH'], f'{task_type} must be one of [TEXT, TEXT_BATCH]'
        target_function = self.map_task2function[task_type]
        response = target_function(encoded_message)
        return response