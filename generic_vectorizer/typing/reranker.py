from pydantic import Field, BaseModel, field_validator, ConfigDict 
from typing import Optional

bge_reranker_models = [
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-v2-gemma",
    "BAAI/bge-reranker-v2-minicpm-layerwise"
]

class FlagRerankerConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name_or_path: str = Field(default='BAAI/bge-reranker-v2-m3')
    device: str = Field(default='cpu')
    use_fp16: bool = Field(default=True)
    cache_dir: Optional[str] = Field(default=None)

    @field_validator('model_name_or_path')
    @classmethod
    def validate_model_name(cls, v):
        if v not in bge_reranker_models:
            raise ValueError(f"model_name_or_path must be one of {bge_reranker_models}")
        return v