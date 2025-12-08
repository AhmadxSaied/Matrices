from decimal import Decimal
from dataclasses import dataclass
from typing_extensions import List

@dataclass
class Steps: 
    stepNumber:int
    description:str
    X_U : Decimal | None = None
    X_L : Decimal | None = None
    X_r : Decimal | None = None
    F_Xr : Decimal | None = None
    F_Xl : Decimal | None  = None
    F_Xu : Decimal | None  = None
    Xi_0 : Decimal | None = None
    Xi_1 : Decimal | None = None
    Error : Decimal
    

@dataclass
class Response:
    status : str
    result : Decimal
    executionTime : float
    TotalIterations : int
    steps : List[Steps]
    errorMessage:str

def addsteps(
    all_steps:List['Steps'],
    description: str,
    X_U : Decimal | None = None,
    X_L : Decimal | None = None,
    X_r : Decimal | None = None,
    F_Xr : Decimal | None = None,
    F_Xl : Decimal | None  = None,
    F_Xu : Decimal | None  = None,
    Xi_0 : Decimal | None = None,
    Xi_1 : Decimal | None = None,
    Error : Decimal | None = None
)-> None :
    StepNumber = len(all_steps)
    new_step = Steps(
        stepNumber= StepNumber,
        description= description,
        X_U= X_U,
        X_L = X_L,
        X_r = X_r ,
        F_Xr = F_Xr,
        F_Xl = F_Xl,
        F_Xu = F_Xu,
        Xi_0 = Xi_0,
        Xi_1 = Xi_1,
        Error =  Error
        )
    all_steps.append(new_step)
    
    