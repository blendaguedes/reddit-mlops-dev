from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., description="Texto do post do Reddit")


class PredictResponse(BaseModel):
    prediction: int = Field(
        ..., description="0 = baixo engajamento, 1 = alto engajamento"
    )
    probability: float = Field(..., description="Probabilidade da classe positiva")
    model: str = "logistic_regression"
