from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Embedding(_message.Message):
    __slots__ = ("dense_values", "sparse_values")
    class SparseValuesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    DENSE_VALUES_FIELD_NUMBER: _ClassVar[int]
    SPARSE_VALUES_FIELD_NUMBER: _ClassVar[int]
    dense_values: _containers.RepeatedScalarFieldContainer[float]
    sparse_values: _containers.ScalarMap[str, float]
    def __init__(self, dense_values: _Optional[_Iterable[float]] = ..., sparse_values: _Optional[_Mapping[str, float]] = ...) -> None: ...

class TextEmbeddingRequest(_message.Message):
    __slots__ = ("embed_strategy", "target_topic", "text", "chunk_size", "return_dense", "return_sparse")
    EMBED_STRATEGY_FIELD_NUMBER: _ClassVar[int]
    TARGET_TOPIC_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
    RETURN_DENSE_FIELD_NUMBER: _ClassVar[int]
    RETURN_SPARSE_FIELD_NUMBER: _ClassVar[int]
    embed_strategy: str
    target_topic: str
    text: str
    chunk_size: int
    return_dense: bool
    return_sparse: bool
    def __init__(self, embed_strategy: _Optional[str] = ..., target_topic: _Optional[str] = ..., text: _Optional[str] = ..., chunk_size: _Optional[int] = ..., return_dense: bool = ..., return_sparse: bool = ...) -> None: ...

class TextEmbeddingResponse(_message.Message):
    __slots__ = ("status", "error", "embedding")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    status: bool
    error: str
    embedding: Embedding
    def __init__(self, status: bool = ..., error: _Optional[str] = ..., embedding: _Optional[_Union[Embedding, _Mapping]] = ...) -> None: ...

class TextBatchEmbeddingRequest(_message.Message):
    __slots__ = ("embed_strategy", "target_topic", "texts", "chunk_size", "return_dense", "return_sparse")
    EMBED_STRATEGY_FIELD_NUMBER: _ClassVar[int]
    TARGET_TOPIC_FIELD_NUMBER: _ClassVar[int]
    TEXTS_FIELD_NUMBER: _ClassVar[int]
    CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
    RETURN_DENSE_FIELD_NUMBER: _ClassVar[int]
    RETURN_SPARSE_FIELD_NUMBER: _ClassVar[int]
    embed_strategy: str
    target_topic: str
    texts: _containers.RepeatedScalarFieldContainer[str]
    chunk_size: int
    return_dense: bool
    return_sparse: bool
    def __init__(self, embed_strategy: _Optional[str] = ..., target_topic: _Optional[str] = ..., texts: _Optional[_Iterable[str]] = ..., chunk_size: _Optional[int] = ..., return_dense: bool = ..., return_sparse: bool = ...) -> None: ...

class TextBatchEmbeddingResponse(_message.Message):
    __slots__ = ("status", "error", "embeddings")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    status: bool
    error: str
    embeddings: _containers.RepeatedCompositeFieldContainer[Embedding]
    def __init__(self, status: bool = ..., error: _Optional[str] = ..., embeddings: _Optional[_Iterable[_Union[Embedding, _Mapping]]] = ...) -> None: ...

class TextRerankScoresRequest(_message.Message):
    __slots__ = ("query", "target_topic", "corpus", "normalize")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    TARGET_TOPIC_FIELD_NUMBER: _ClassVar[int]
    CORPUS_FIELD_NUMBER: _ClassVar[int]
    NORMALIZE_FIELD_NUMBER: _ClassVar[int]
    query: str
    target_topic: str
    corpus: _containers.RepeatedScalarFieldContainer[str]
    normalize: bool
    def __init__(self, query: _Optional[str] = ..., target_topic: _Optional[str] = ..., corpus: _Optional[_Iterable[str]] = ..., normalize: bool = ...) -> None: ...

class TextRerankScoresResponse(_message.Message):
    __slots__ = ("status", "error", "scores")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    status: bool
    error: str
    scores: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, status: bool = ..., error: _Optional[str] = ..., scores: _Optional[_Iterable[float]] = ...) -> None: ...
