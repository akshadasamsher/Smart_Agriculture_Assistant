from pydantic import BaseModel, Field


class SoilInput(BaseModel):
    N: float = Field(..., description="Nitrogen ratio in soil")
    P: float = Field(..., description="Phosphorous ratio in soil")
    K: float = Field(..., description="Potassium ratio in soil")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Relative humidity in %")
    ph: float = Field(..., description="Soil pH value")
    rainfall: float = Field(..., description="Rainfall in mm")

    class Config:
        json_schema_extra = {
            "example": {
                "N": 90, "P": 42, "K": 43,
                "temperature": 20.9, "humidity": 82.0,
                "ph": 6.5, "rainfall": 202.9,
            }
        }


class CropPrediction(BaseModel):
    crop: str
    confidence: float
    top_k: list[dict] | None = None


class DiseasePrediction(BaseModel):
    label: str
    plant: str
    condition: str
    healthy: bool
    confidence: float
    low_confidence: bool = False
    top_k: list[dict]
