from pydantic import BaseModel
from typing_extensions import List
from dataclasses import dataclass
from decimal import Decimal
@dataclass
class Input(BaseModel):
    X_Lower : Decimal | None = None
    X_Upper : Decimal | None = None
    percision : int | None = 10
    Tolerance : Decimal | None = Decimal("1e-6")
    Function : str
    Xo_Initial : Decimal | None = None
    X1_Initial : Decimal | None = None
    max_itr : int | None = 50