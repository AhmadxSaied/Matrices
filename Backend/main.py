import time

from pydantic import BaseModel
from typing_extensions import List
from dataclasses import dataclass
from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy import sympify, Symbol, lambdify
import pandas as pd
from response_phase2 import Response,Steps,addsteps
@dataclass
class Item(BaseModel):
    Function : str
    MethodId:str
    X_Lower : Decimal | None = None
    X_Upper : Decimal | None = None
    percision : int | None = 10
    Tolerance : Decimal | None = Decimal("1e-6")
    Xo_Initial : Decimal | None = None
    X1_Initial : Decimal | None = None
    maxIteration : int | None = 50
    
@dataclass
class Plotter(BaseModel):
    LowerBound : float
    UpperBound : float
    NumberOfPoints : int
    Function : str

'''
    Function For Plotting recieved Expression
    Takes in Function ,LowerBound, UpperBound,NumberOfPoints
    returns Fig to be plotted
'''
def plotter_Function(plotter:Plotter):
    True_Lower = min(plotter.LowerBound,plotter.UpperBound)
    True_Upper = max(plotter.LowerBound,plotter.UpperBound)
    assert plotter.NumberOfPoints > 0
    
    Symp_Function = sp.sympify(plotter.Function)
    print((list(Symp_Function.free_symbols)))
    Symbols = list(Symp_Function.free_symbols)
    
    print(Symp_Function)
    MathExpression = sp.lambdify(Symbols,Symp_Function)
    
    LineSpace = np.linspace(True_Lower,True_Upper,plotter.NumberOfPoints)
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

def False_Position(item:Item,all_steps:List['Steps']):
    start_time = time.perf_counter()
    Sp_Function = sp.sympify(item.Function)
    Symbols = list(Sp_Function.free_symbols)
    MathExpression = sp.lambdify(Symbols,Sp_Function)
    assert MathExpression(item.X_Lower) * MathExpression(item.X_Upper) <0
    Error = 1.0
    Xu_Loop = item.X_Lower
    Xl_Loop = item.X_Upper
    iterations= 0
    Xr_Old = 0
    Xrs = []
    Xus=[]
    Xls=[]
    Es = []
    Fxrs =[]
    while(np.abs(Error)>=item.Tolerance and iterations<=item.maxIteration):
        iterations=iterations+1
        FXl = MathExpression(Xl_Loop)
        FXu =MathExpression(Xu_Loop)
        Xr_Loop = ((Xl_Loop * FXu)-(Xu_Loop * FXl))/(FXu - FXl)
        FXr = MathExpression(Xr_Loop)
        addsteps(all_steps,f"Xr = (({Xl_Loop} * {FXu})-({Xu_Loop} * {FXl}))/({FXu} - {FXl})",X_U=Xu_Loop,X_L=Xl_Loop,X_r=Xr_Loop,F_Xr=FXr,F_Xl=FXl,F_Xu=FXu,Error=Error)
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
    if (np.abs(Error) > item.Tolerance):
        res_message = "Error"
        why = "failed to converge with max iteration"
    else:
        res_message = "SUCCESS"
        why = ""
    end_time = time.perf_counter()
    return Response(res_message,Xrs[-1],round(end_time - start_time,6),iterations,all_steps,why)

"""
    Bisection
    Takes in Xlower , Xupper , Tolerance and MaxIterations
    Panda will be Substituted by responsse Class
"""

import pandas as pd
import numpy as np
def Bisection(item:Item,all_steps:List['Steps']):
    start_time = time.perf_counter()
    Sp_Function = sp.sympify(item.Function)
    Symbols = list(Sp_Function.free_symbols)
    MathExpression = sp.lambdify(Symbols,Sp_Function)
    assert MathExpression(item.X_Lower) * MathExpression(item.X_Upper) <0
    Error = 1.0
    Xu_Loop = item.X_Lower
    Xl_Loop = item.X_Upper
    iterations= 0
    Xr_Old = 0
    Xrs = []
    Xus=[]
    Xls=[]
    Es = []
    Fxrs =[]
    while((np.abs(Error)>=item.Tolerance) and (iterations)<=item.maxIteration):
        iterations = iterations+1
        FXl = MathExpression(Xl_Loop)
        Xr_Loop = Xu_Loop - (Xu_Loop-Xl_Loop)/2
        FXr = MathExpression(Xr_Loop)
        Decision = (FXl * FXr)
        addsteps(all_steps,f"Xr = ({Xu_Loop+Xl_Loop})/2",Xu_Loop,Xl_Loop,X_r=Xr_Loop,F_Xr=FXr,F_Xl=FXl,Error=Error)
        if(Decision < 0):
            Xu_Loop = Xr_Loop
        elif(Decision > 0):
            Xl_Loop = Xr_Loop
        else:
            end_time = time.perf_counter()
            return Response("SUCCESS",Xr_Old,round(end_time - start_time,6),iterations,all_steps,"")

        Error = (Xr_Loop -Xr_Old)/(Xr_Loop)
        Xr_Old = Xr_Loop
        Xrs.append(Xr_Loop)
        Xus.append(Xu_Loop)
        Xls.append(Xl_Loop)
        Es.append(Error)
        Fxrs.append(FXr)
    if(np.abs(Error) > item.Tolerance):
        res_message = "Error"
        why = "failed to converge with max iteration"
    else:
        res_message = "SUCCESS"
        why = ""
    end_time = time.perf_counter()
    return Response(res_message,Xr_Old,round(end_time - start_time,6),iterations,all_steps,why)


def str_to_func(exp):
    x = Symbol('x')
    sym_expr = sympify(exp)
    return lambdify(x, sym_expr, 'math')  # or "numpy"

# ! secant method
# @para
# p0 : 1st initial guess
# p1 : 2nd initial guess
# tol : tolerance
# max : max number of itrations
# exp : f(x) in str format
# per : percision
def secant_method(item:Item,all_steps:List['Steps']):
    start_time = time.perf_counter()
    f = str_to_func(item.Function)
    itration = 0
    rel_error = 100
    p0 = item.Xo_Initial
    p1 = item.X1_Initial
    while itration != item.maxIteration and rel_error > item.Tolerance:
        fp0 = f(p0)
        fp1 = f(p1)
        new_p = p1 - (fp1*(p1-p0)/(fp1-fp0))
        rel_error = abs((new_p - p1)/new_p) * 100
        addsteps(all_steps,f"X{itration+2} = {p1} - (({fp1}*({p1}-{p0})/({fp1}-{fp0}))) ",Error=rel_error,X_r=new_p,X_U=p1,X_L=p0)
        itration += 1
        # print(f"itration {itration}\nnew_p {new_p}\nrel_error {rel_error}")
        p0 = p1
        p1 = new_p
    if(rel_error > item.Tolerance):
        res_message = "Error"
        why = "failed to converge with max iteration"
    else:
        res_message = "SUCCESS"
        why = ""
    end_time = time.perf_counter()
    return Response(res_message,p1,round(end_time-start_time,6),itration,all_steps,why)





# ! fixed point itration method
# @para
# x : initial guess of fixed point
# tol : tolerance
# max : max number of itrations
# exp : magic function g(x) in str format
# per : percision
def fixed_point_method(item:Item,all_steps:List['Steps']):
    start_time = time.perf_counter()
    magic_function = str_to_func(item.Function)
    itration = 0
    rel_error = 100
    x=item.Xo_Initial
    tol = item.Tolerance
    max_itr=item.maxIteration
    while itration != max_itr and rel_error > tol:
        new_x = magic_function(x)
        rel_error = abs((new_x - x)/new_x) * 100
        itration += 1
        addsteps(all_steps,f"X{itration} = {new_x}",Error=rel_error,Xi_0=x,X_r=new_x)
        print(f"itration {itration}\nnew_p {new_x}\nrel_error {rel_error}")
        x = new_x
    if(rel_error > item.Tolerance):
        res_message = "Error"
        why = "failed to converge with max iteration"
    else:
        res_message = "SUCCESS"
        why = ""
    end_time = time.perf_counter()
    return Response(res_message,x,round(end_time-start_time,6),itration,all_steps,why)

