from dataclasses import dataclass
from typing import List, Union, Dict, Any
from pydantic import BaseModel

@dataclass
class Input(BaseModel):
    matrixA: List[List[float]] 
    matrixB: List[float]
    methodID: str
    precision: int | None = 10
    methodParams: Dict[str, Any]
    max_iterations:int
    tolerance:float
