from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Literal

bge_m3_models = [
    "BAAI/bge-m3",
    "BAAI/bge-m3-unsupervised",
    "BAAI/bge-m3-retromae",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5"
]

class BGEM3FlagModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name_or_path: str = Field(default='BAAI/bge-m3')
    device: str = Field(default='cpu')
    use_fp16: bool = Field(default=True)
    pooling_method: Literal['cls', 'mean'] = Field(default='cls')

    @field_validator('model_name_or_path')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if v not in bge_m3_models:
            raise ValueError(f"model_name_or_path must be one of {bge_m3_models}")
        return v

    @field_validator('pooling_method')
    @classmethod
    def validate_pooling_method(cls, v: str) -> str:
        if v not in ['cls', 'mean']:
            raise ValueError("pooling_method must be either 'cls' or 'mean'")
        return v