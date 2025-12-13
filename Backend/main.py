import base64
import io
import time

from pydantic import BaseModel
from sympy.parsing.mathematica import *
from sympy.parsing.mathematica import parse_mathematica
from sympy.parsing.sympy_parser import transformations, implicit_multiplication_application
from typing_extensions import List
from dataclasses import dataclass
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy import sympify, Symbol, lambdify
import pandas as pd
from response_phase2 import Response,Steps,addsteps
from autograd import grad
@dataclass
class Item(BaseModel):
    Function : str
    MethodId:str
    X_Lower : Decimal | None = None
    X_Upper : Decimal | None = None
    precision : int | None = 10
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


    Symbols = list(Symp_Function.free_symbols)
    if not Symbols:
        return "error"

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

    buf = io.BytesIO()
    plt.savefig(buf, format='png',bbox_inches='tight')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    data_url = f"data:image/png;base64,{image_base64}"
    return data_url

"""
    FalsePosition method
    Takes in Xl,Xu , Tolerance and MaxIterations and Function to Calculate
    Panda Table will be substituted by response class
"""

def False_Position(item:Item,all_steps:List['Steps']):
    start_time = time.perf_counter()
    getcontext().prec = item.precision if item.precision is not None else 10
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
    return Response(res_message,Xrs[-1],round(end_time - start_time,6),iterations,all_steps,why,FinalError=Error)

"""
    Bisection
    Takes in Xlower , Xupper , Tolerance and MaxIterations
    Panda will be Substituted by responsse Class
"""

import pandas as pd
import numpy as np
def Bisection(item:Item,all_steps:List['Steps']):
    getcontext().prec = item.precision if item.precision is not None else 10
    start_time = time.perf_counter()
    Sp_Function = sp.sympify(item.Function)
    Symbols = list(Sp_Function.free_symbols)
    MathExpression = sp.lambdify(Symbols,Sp_Function)
    assert MathExpression(Decimal(str(item.X_Lower))) * MathExpression(Decimal(str(item.X_Upper))) <0
    Error = Decimal("1.0")
    Xu_Loop = Decimal(str(item.X_Lower))
    Xl_Loop = Decimal(str(item.X_Upper))
    iterations= 0
    Xr_Old = Decimal('0')
    Xrs = []
    Xus=[]
    Xls=[]
    Es = []
    Fxrs =[]
    Xr_Loop=0
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
            Error =0
            break

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
    return Response(res_message,Xr_Loop,round(end_time - start_time,6),iterations,all_steps,why,FinalError=Error)


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
    getcontext().prec = item.precision if item.precision is not None else 10
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
    return Response(res_message,p1,round(end_time-start_time,6),itration,all_steps,why,FinalError=rel_error)





# ! fixed point itration method
# @para
# x : initial guess of fixed point
# tol : tolerance
# max : max number of itrations
# exp : magic function g(x) in str format
# per : percision
def fixed_point_method(item:Item,all_steps:List['Steps']):
    getcontext().prec = item.precision if item.precision is not None else 10
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
    return Response(res_message,x,round(end_time-start_time,6),itration,all_steps,why,FinalError=rel_error)

def Newton_Normal(item: Item,all_steps:List['Steps']):
    getcontext().prec = item.percision if item.percision is not None else 10
    start_time = time.perf_counter()
    initial_guess = Decimal(item.Xo_Initial) + 0
    maxIterations = item.maxIteration
    x = [initial_guess]
    oldX = initial_guess
    f = str_to_func(item.Function)
    f_p = grad(f)
    
    def app(x):
        return x -f(x) / Decimal(f_p(float(x))) 
    iterations = 1;
    
    for i in range(maxIterations):
        if(f_p(float(oldX)) == 0):
            return Response(status="Failed", errorMessage="The value of the f`(x) at xo initial = zero try another initial guess")
        newX = app(oldX)
        x.append(newX)
        ea = abs(((newX - oldX)/newX) * 100);
        addsteps(all_steps=all_steps,description=f"{i+1}. current approximation = {newX}" , X_r=newX, F_Xr=f(newX), Error=ea, )
        if(ea < item.Tolerance):
            print(f"epsilon reached in {iterations} iterations")
            end_time = time.perf_counter()
            return Response(status="success", result=newX, executionTime=round(end_time - start_time, 6), TotalIterations=iterations, steps=all_steps, FinalError=ea, errorMessage=None )
        oldX = newX
        iterations +=1
    end_time = time.perf_counter()
    print("max iterations reached")
    return Response(status="failed", result=newX, executionTime=round(end_time - start_time, 6), TotalIterations=iterations, steps=all_steps, FinalError=ea, errorMessage="Max iteration reached" ) 

def Newton_modified(item: Item,all_steps: List['Steps']):
    getcontext().prec = item.percision if item.percision is not None else 10
    start_time = time.perf_counter()
    initial_guess = Decimal(item.Xo_Initial) + 0
    x = [initial_guess]
    oldX = initial_guess
    f = str_to_func(item.Function)
    f_p = grad(f)
    f_pp = grad(f_p)
    def app(x):
        f_x = Decimal(f(float(x)))
        fp_x = Decimal(f_p(float(x)))
        fpp_x = Decimal(f_pp(float(x)))
        return Decimal(x - (f_x * fp_x)/ (fp_x**2 - f_x * fpp_x))
    iterations = 1;
    
    for i in range(item.maxIteration):
        if(f_p(float(oldX)) == 0):
            return Response(status="Failed", errorMessage="The value of the f`(x) at xo initial = zero, try another initial guess")
        newX = app(oldX)
        newX = app(oldX);
        x.append(newX);
        ea = abs(((newX - oldX)/newX) * 100);
        addsteps(all_steps=all_steps,description=f"{i+1}. current approximation = {newX}" , X_r=newX, F_Xr=f(newX), Error=ea, )
        if(ea < item.Tolerance):
            print(f"epsilon reached in {iterations} iterations")
            end_time = time.perf_counter()
            return Response(status="success", result=newX, executionTime=round(end_time - start_time, 6), TotalIterations=iterations, steps=all_steps, FinalError=ea, errorMessage=None )
        oldX = newX
        iterations +=1
    print("max iterations reached")
    end_time = time.perf_counter()
    return Response(status="failed", result=newX, executionTime=round(end_time - start_time, 6), TotalIterations=iterations, steps=all_steps, FinalError=ea, errorMessage="Max iteration reached" )
