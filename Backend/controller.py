from response_phase2 import Response
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import main
origin = ["http://localhost:4200"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = origin,
    allow_methods = ['*'],
    allow_headers = ['*']
)

def Map_to_Function(request_data : main.Item):
    MethodId = request_data.MethodId
    List_Steps = []
    if MethodId == "Bisection":
        Response = main.Bisection(request_data,List_Steps)
        return Response
    if MethodId == "False_Position":
        Response = main.False_Position(request_data,List_Steps)
        return Response
    if MethodId == "Newton_Normal":
        Response = main.Newton_Normal(request_data,List_Steps)
        return Response
    if MethodId == "Newton_modified":
        Response = main.Newton_modified(request_data,List_Steps)
        return Response
    if MethodId == "Secant":
        Response = main.secant_method(request_data,List_Steps)
        return Response
    if MethodId == "FixedPoint":
        Response = main.fixed_point_method(request_data,List_Steps)
        return Response
    return None
@app.post("/solve_root")
async def Phase2(request_data : main.Item):
    response = Map_to_Function(request_data)
    if(response == None):
        return "failed"
    else:
        return response


@app.post("/plot")
async def Map(request_data:main.Plotter):
    return main.plotter_Function(request_data)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)