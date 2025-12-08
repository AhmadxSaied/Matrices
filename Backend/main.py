from pydantic import BaseModel
from typing_extensions import List
from dataclasses import dataclass
from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import pandas as pd
@dataclass
class Item(BaseModel):
    X_Lower : Decimal | None = None
    X_Upper : Decimal | None = None
    percision : int | None = 10
    Tolerance : Decimal | None = Decimal("1e-6")
    Function : str
    Xo_Initial : Decimal | None = None
    X1_Initial : Decimal | None = None
    max_itr : int | None = 50
    
    

'''
    Function For Plotting recieved Expression
    Takes in Function ,LowerBound, UpperBound,NumberOfPoints
    returns Fig to be plotted
'''
def plotter_Function(Function:str,LowerBound:float,UpperBound:float,NumberOfPoints:int):
    True_Lower = min(LowerBound,UpperBound)
    True_Upper = max(LowerBound,UpperBound)
    assert NumberOfPoints > 0
    
    Symp_Function = sp.sympify(Function)
    print((list(Symp_Function.free_symbols)))
    Symbols = list(Symp_Function.free_symbols)
    
    print(Symp_Function)
    MathExpression = sp.lambdify(Symbols,Symp_Function)
    
    LineSpace = np.linspace(True_Lower,True_Upper,NumberOfPoints)
    Results = MathExpression(LineSpace)
    
    Fig, ax = plt.subplots(figsize=(15,10))
    ax.set_xlabel("Xaxis")
    ax.set_ylabel("Yaxis")
    ax.set_title(f"Function Plotter: F(x) = {Symp_Function}")
    ax.plot(LineSpace,Results)
    ax.axhline(y=0,color="black",linestyle="solid")
    ax.axvline(x=0,color="black",linestyle="solid")
    ax.grid(True)
    return Fig,ax

"""
    FalsePosition method
    Takes in Xl,Xu , Tolerance and MaxIterations and Function to Calculate
    Panda Table will be substituted by response class
"""

def False_Position(Xl:float,Xu:float,Tolerance:float,maxIteration:int,Function : str):
    Sp_Function = sp.sympify(Function)
    Symbols = list(Sp_Function.free_symbols)
    MathExpression = sp.lambdify(Symbols,Sp_Function)
    assert MathExpression(Xl) * MathExpression(Xu) <0
    Error = 1.0
    Xu_Loop = Xu
    Xl_Loop = Xl
    iterations= 0
    Xr_Old = 0
    Xrs = []
    Xus=[]
    Xls=[]
    Es = []
    Fxrs =[]
    while(np.abs(Error)>=Tolerance and iterations<=maxIteration):
        iterations=iterations+1
        FXl = MathExpression(Xl_Loop)
        FXu =MathExpression(Xu_Loop)
        Xr_Loop = ((Xl_Loop * FXu)-(Xu_Loop * FXl))/(FXu - FXl)
        FXr = MathExpression(Xr_Loop)
        if(np.sign(FXu)==np.sign(FXr)):
            Xu_Loop = Xr_Loop
        elif(np.sign(FXl)==np.sign(FXr)):
            Xl_Loop = Xr_Loop
        Error = (Xr_Loop - Xr_Old)/(Xr_Loop)
        Xr_Old = Xr_Loop
        Xrs.append(Xr_Loop)
        Xus.append(Xu_Loop)
        Xls.append(Xl_Loop)
        Es.append(Error)
        Fxrs.append(FXr)
    table = pd.DataFrame({"Xr":Xrs,"Xu":Xus,"Xl":Xls,"Error":Es,"Fxr":Fxrs})
    return table

"""
    Bisection
    Takes in Xlower , Xupper , Tolerance and MaxIterations
    Panda will be Substituted by responsse Class
"""

import pandas as pd
import numpy as np
def Bisection(Xl:float,Xu:float,Tolerance:float,maxIteration:int,Function:str):
    Sp_Function = sp.sympify(Function)
    Symbols = list(Sp_Function.free_symbols)
    MathExpression = sp.lambdify(Symbols,Sp_Function)
    assert MathExpression(Xl) * MathExpression(Xu) <0
    Error = 1.0
    Xu_Loop = Xu
    Xl_Loop = Xl
    iterations= 0
    Xr_Old = 0
    Xrs = []
    Xus=[]
    Xls=[]
    Es = []
    Fxrs =[]
    while((np.abs(Error)>=Tolerance) and (iterations)<=maxIteration):
        iterations = iterations+1
        FXl = MathExpression(Xl)
        Xr_Loop = Xu_Loop - (Xu_Loop-Xl_Loop)/2
        FXr = MathExpression(Xr_Loop)
        Decision = (FXl * FXr)
        if(Decision < 0):
            Xu_Loop = Xr_Loop
        elif(Decision > 0):
            Xl_Loop = Xr_Loop
        else:
            break
        Error = (Xr_Loop -Xr_Old)/(Xr_Loop)
        Xr_Old = Xr_Loop
        Xrs.append(Xr_Loop)
        Xus.append(Xu_Loop)
        Xls.append(Xl_Loop)
        Es.append(Error)
        Fxrs.append(FXr)
    table = pd.DataFrame({"Xr":Xrs,"Xu":Xus,"Xl":Xls,"Error":Es,"Fxr":Fxrs})
    return table