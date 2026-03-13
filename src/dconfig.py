
from pydantic import BaseModel
#from pydantic_settings import SettingsConfigDict, BaseSettings
from enum import Enum


class Chunkerconfig(BaseModel):
    max_tokens:int = 512
    overlap: int = 16
    
class EmbeddingsConfig(BaseModel):
    model_name:str  = "BAAI/bge-base-en-v1.5"
    dims: int       = 768
    batch_size:int  = 32
    