from dataclasses import dataclass
from typing import List, Union, Dict, Any
from pydantic import BaseModel
from decimal import Decimal, getcontext
@dataclass
class Input(BaseModel):
    matrix: List[List[Decimal]]
    methodID: str
    precision: int | None = 10
    methodParams: Dict[str, Any]
    size: int
    vector_of_sol: list[Decimal]
    initial_guess: list[Decimal] = []
    Tolerance:int |None= Decimal("1e-6")