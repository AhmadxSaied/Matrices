import decimal
from dataclasses import dataclass
from typing import List,Dict,Union
import copy
from decimal import Decimal, getcontext

@dataclass
class Steps:
    stepNumber:int
    description: str
    matrixA: List[List[Decimal]]
    matrixB:List[Decimal]
    L:List[List[Decimal]]
    U:List[List[Decimal]]

@dataclass
class Response:
    status: str 
    results: List[Decimal]
    executionTime: float
    TotalIternation: int
    steps: List[Steps]
    errorMessage:str

def addsteps(
        all_steps:List['Steps'],
        description:str,
        matrix:List[List[Decimal]],
        vector:List[Decimal],
        L:List[List[Decimal]] | None = None,
        U:List[List[Decimal]] | None = None,
) -> None:
    matrix_copy = copy.deepcopy(matrix)
    vector_copy = vector[:]
    Step_number = len(all_steps)+1
    new_step = Steps(
        stepNumber=Step_number,
        description=description,
        matrixA=matrix_copy,
        matrixB=vector_copy,
        L = L,
        U = U
    )
    all_steps.append(new_step)