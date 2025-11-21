from input import Input
from response import Response
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import main
origin=["http://localhost:4200"]
methods=["Gauss_"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origin,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)
def Map_to_Function(request_data : main.Item):
    MethodId = request_data.MethodId
    List_Steps = []
    if MethodId == "Naive_Gauss":
        Response = main.Naive_gauss_elimination(request_data,List_Steps)
        return Response
    if MethodId == "Gauss_elimination_Pivot":
        Response = main.Gauss_elimination_with_partial_pivoting(request_data,List_Steps)
        return Response
    if MethodId == "Gauss_elimination_Pivoting_Scaling":
        Response = main.Gauss_elimination_with_partial_pivoting_and_scaling(request_data,List_Steps)
        return Response
    if MethodId == "Gauss_Jordan":
        Response = main.Gauss_Jordan_elimination(request_data,List_Steps)
        return Response
    if MethodId == "Gauss_Seidel":
        Response = main.Gauss_Seidel_method(request_data,List_Steps)
        return Response
    if MethodId == "Jacobi":
        Response = main.Jacobi_method(request_data,List_Steps)
        return Response
    if MethodId == "LU_decomposition_Doolittle":
        Response = main.LU_decomposition_Doolittle_method(request_data,List_Steps)
        return Response
    if MethodId == "LU_decomposition_Cholesky":
        Response = main.LU_decomposition_Cholesky_method(request_data,List_Steps)
        return Response
    return None
    
        
@app.post("/solve")
async def slove_matrix(request_data: main.Item) -> Response:
    Response = Map_to_Function(request_data=request_data)
    if(Response is not None):
        return Response
    else:
        return "Error Occured Check the Solve endPoint"
    
if __name__ == "__main__":
    uvicorn.run(app=app,port=8000)