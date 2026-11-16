"""Pydantic schemas for the Indic Climate Stance API."""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class TokenAttribution(BaseModel):
    token: str
    attribution: float


class PredictRequest(BaseModel):
    text: str = Field(..., description="Input text in Hindi, Bengali, Hinglish, Banglish, or English")
    explain: bool = Field(False, description="Whether to compute Integrated Gradients token-level attributions")


class PredictResponse(BaseModel):
    text: str
    clean_text: Optional[str] = None
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    language: Optional[str] = None
    script: Optional[str] = None
    is_code_mixed: bool = False
    has_negation: bool = False
    truncated: bool = False
    latency_ms: float = 0.0
    device: str = "cpu"
    explanations: Optional[List[TokenAttribution]] = None
    error: Optional[str] = None


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., description="List of social media texts to classify")
    explain: bool = Field(False, description="Whether to compute explanations for each prediction")


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total: int
    latency_ms: float
    throughput_items_per_sec: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    checkpoint: str
    device: str
    version: str


class ModelInfoResponse(BaseModel):
    name: str
    architecture: str
    base_encoder: str
    layers: int
    pooling: str
    max_sequence_length: int
    lora_rank: int
    lora_alpha: int
    total_parameters: int
    trainable_parameters: int
    classes: List[str]
    target_claim: str
    supported_languages: List[str]
    supported_scripts: List[str]
    distillation_teacher: str
    distillation_hyperparameters: Dict[str, Any]

