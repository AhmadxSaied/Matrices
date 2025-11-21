from dataclasses import dataclass
from typing import List,Dict,Union
import copy

@dataclass
class Steps:
    stepNumber:int
    description: str
    matrixA: List[List[float]]
    matrixB:List[float]
    L:List[List[float]]
    U:List[List[float]]

@dataclass
class Response:
    status: str 
    results: List[float]
    executionTime: float
    TotalIternation: int
    steps: List[Steps]
    errorMessage:str

def addsteps(
        all_steps:List['Steps'],
        description:str,
        matrix:List[List[float]],
        vector:List[float]
) -> None:
    matrix_copy = copy.deepcopy(matrix)
    vector_copy = vector[:]
    Step_number = len(all_steps)+1
    new_step = Steps(
        stepNumber=Step_number,
        description=description,
        matrixA=matrix_copy,
        matrixB=vector_copy
    )
    all_steps.append(new_step)