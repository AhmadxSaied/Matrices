import decimal
from dataclasses import dataclass
from typing_extensions import List,TypedDict,Union
import copy
from decimal import Decimal, getcontext
class PivotIndex(TypedDict):
    r:int
    c:int

@dataclass
class Steps:
    stepNumber:int
    description: str
    matrixA: List[List[Decimal]]
    matrixB:List[Decimal]
    L:List[List[Decimal]]
    U:List[List[Decimal]]
    Error:Decimal
    pivotIndex:PivotIndex | None = None
    highlightRow:int | None = None

@dataclass
class Response:
    status: str 
    results: List[Decimal]
    executionTime: float
    TotalIternation: int
    steps: List[Steps]
    errorMessage:str
    L:List[List[Decimal]] | None = None
    U:List[List[Decimal]] | None =None
    equations:List[str] | None = None
    Diagonal:bool | None = None
    pivotIndex:PivotIndex | None = None
    highlightRow:int | None = None

def addsteps(
        all_steps:List['Steps'],
        description:str,
        matrix:List[List[Decimal]],
        vector:List[Decimal],
        L:List[List[Decimal]] | None = None,
        U:List[List[Decimal]] | None = None,
        Error:Decimal | None = None,
        pivotIndex:PivotIndex | None = None,
        highlightRow:int | None = None
) -> None:
    matrix_copy = copy.deepcopy(matrix)
    vector_copy = vector[:]
    Step_number = len(all_steps)+1
    new_step = Steps(
        stepNumber=Step_number,
        description=description,
        matrixA=matrix_copy,
        matrixB=vector_copy,
        L = copy.deepcopy(L),
        U = copy.deepcopy(U),
        Error = copy.deepcopy(Error),
        pivotIndex = pivotIndex,
        highlightRow = highlightRow

    )
    all_steps.append(new_step)