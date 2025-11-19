from input import Input
from response import Response
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import main
origin=["https/localhost:4200"]
methods=["Gauss_"]
app = FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=origin,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)
@app.post("/solve")
async def slove_matrix(request_data: Input) -> Response:
    method = request_data.methodID