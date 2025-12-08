from response_phase2 import Response
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import main

origin = ["http//localhost:4200"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origin = origin,
    allow_method = ['*'],
    allow_headers = ['*']
)

def Map_to_Function(request_data : main.Item):
    MethodId = request_data.MethodId
    List_Steps = []
    if MethodId == "Bisection":
        Response = main.Bisection(request_data,List_Steps)
        return Response
    if MethodId == "False_Regula":
        Response = main.False_Regula(request_data,List_Steps)
        return Response
    if MethodId == "Newton_Normal":
        Response = main.Newton_Normal(request_data,List_Steps)
        return Response
    if MethodId == "Newton_modified":
        Response = main.Newton_modified(request_data,List_Steps)
        return Response
    if MethodId == "Secant":
        Response = main.Secant(request_data,List_Steps)
        return Response
    if MethodId == "FixedPoint":
        Response = main.FixedPoint(request_data,List_Steps)
        return Response
    return None