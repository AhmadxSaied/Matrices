from decimal import Decimal, getcontext
from pydantic import BaseModel
import time
from response import Response,Steps,addsteps
import helper
from typing import List
import copy

# Defines the structure of the JSON data we expect to receive.
class Item(BaseModel):
    MethodId: str
    precision: int | None = 10
    size: int
    matrix: list[list[Decimal]]
    vector_of_sol: list[Decimal]
    initial_guess: list[Decimal] = []
    max_iterations:int
    Tolerance : Decimal | None = Decimal("1e-5")

# +++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++ methods ++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++
# ! Naive gauss elimination
def Naive_gauss_elimination(item: Item,all_steps:List['Steps']):
    print(item)
    start_time = time.perf_counter()
    A_copy = copy.deepcopy(item.matrix)
    B_copy = item.vector_of_sol[:]
    getcontext().prec = item.precision if item.precision is not None else 10
    if helper.forward_elimination_withoutPivoting(item.size, A_copy, B_copy,all_steps) == "error":
        end_time = time.perf_counter()
        return Response("Error",[],round(end_time-start_time,6),0,all_steps,"Singluar matrix or division by zero ")
    status = helper.check_havesol(item.size,item.vector_of_sol,item.matrix,all_steps)
    print(item.matrix)
    if(status != "unique"):
        end_time = time.perf_counter()
        if(status == "None"):
            return Response("Error",[],round(end_time-start_time,6),0,all_steps,"the system has no solution")
        else:
            return Response("Error",[],round(end_time-start_time,6),0,all_steps,"the system has Infinite number of solution")
    print(item.matrix)
    vector_of_unknowns = helper.backward_substitution(item.size, A_copy,B_copy,all_steps)
    end_time = time.perf_counter()
    return Response("SUCCESS",vector_of_unknowns,round(end_time - start_time,6),0,all_steps,"")

# ! Gauss Elimination with Partial Pivoting
def Gauss_elimination_with_partial_pivoting(item: Item,all_steps:List['Steps']):
    start_time = time.perf_counter()
    A_copy = copy.deepcopy(item.matrix)
    B_copy = item.vector_of_sol[:]
    getcontext().prec = item.precision if item.precision is not None else 10
    if helper.forward_elimination_withPivoting(item.size, A_copy, B_copy,all_steps) == "error":
        end_time = time.perf_counter()
        return Response("Error",[],round(end_time-start_time,6),0,all_steps,"Singluar matrix")
    status = helper.check_havesol(item.size,A_copy,B_copy,all_steps)
    if(status != "unique"):
        end_time = time.perf_counter()
        if(status == "None"):
            return Response("Error",[],round(end_time-start_time,6),0,all_steps,"the system has no solution")
        else:
            return Response("Error",[],round(end_time-start_time,6),0,all_steps,"the system has Infinite number of solution")
    end_time = time.perf_counter()
    vector_of_unknowns = helper.backward_substitution(item.size, A_copy, B_copy,all_steps)
    return Response("SUCCESS",vector_of_unknowns,round(end_time - start_time,6),0,all_steps,"")

# ! Gauss Elimination with Partial Pivoting and scaling
def Gauss_elimination_with_partial_pivoting_and_scaling(item: Item,all_steps:List['Steps']):
    start_time = time.perf_counter()
    A_copy = copy.deepcopy(item.matrix)
    B_copy = item.vector_of_sol[:]
    getcontext().prec = item.precision if item.precision is not None else 10
    if helper.forward_elimination_withPivoting_and_scaling(item.size, item.matrix, item.vector_of_sol) == "error":
       end_time = time.perf_counter()
       return Response("Error",[],round(end_time-start_time,6),0,all_steps,"Singluar matrix")
    status = helper.check_havesol(item.size,item.vector_of_sol,item.matrix,all_steps)
    if(status != "unique"):
        end_time = time.perf_counter()
        if(status == "None"):
            return Response("Error",[],round(end_time-start_time,6),0,all_steps,"the system has no solution")
        else:
            return Response("Error",[],round(end_time-start_time,6),0,all_steps,"the system has Infinite number of solution")
    vector_of_unknowns = helper.backward_substitution(item.size, item.matrix, item.vector_of_sol)
    return Response("SUCCESS",vector_of_unknowns,round(end_time - start_time,6),0,all_steps,"")

# ! Gauss-Jordan elimination (with partial pivoting)
def Gauss_Jordan_elimination(item: Item,all_steps:List['Steps']):
    getcontext().prec = item.precision if item.precision is not None else 10
    error_status = helper.forward_elimination_withPivoting(item.size,item.matrix,item.vector_of_sol,Steps)
    if error_status == "error":
        return error_status
    helper.backward_substitution(item.size,item.matrix,item.vector_of_sol,Steps)
    return item.vector_of_sol

# ! Gauss-Seidel Method (without pivoting)
def Gauss_Seidel_method(item: Item,all_steps:List['Steps']):
    timer_start = time.perf_counter()
    getcontext().prec = item.precision if item.precision is not None else 10
    values_of_unknowns = [Decimal(str(x)) for x in item.initial_guess]
    # Check for zero pivots (diagonal elements) before starting
    for i in range(item.size):
        if item.matrix[i][i] == 0:
            timer_stop = time.perf_counter()
            addsteps(all_steps,"Cant use Gauss Seidel because the pivot is zero and we cant divide by zero",item.matrix,item.vector_of_sol)
            return Response("Error",item.vector_of_sol,round(timer_stop - timer_start,6),0,all_steps,"can't divide by zero")
    is_diagonally_dominant = helper.check_diagonally_dominant(item.size,item.vector_of_sol,item.matrix,all_steps)
    for k in range(item.max_iterations):
        previous_x = values_of_unknowns[:]
        for row in range(item.size):
            s = Decimal("0")  # s == sum
            for col in range(item.size):
                if col != row:
                    s += values_of_unknowns[col] * item.matrix[row][col]
            values_of_unknowns[row] = (
                item.vector_of_sol[row] - s) / item.matrix[row][row]
        # --- Check for Convergence ---
        # Compute the maximum relative change between the current and previous vector
        max_relative_error = Decimal("0")
        for i in range(item.size):
            # |current - previous| / |current|
            delta = abs(values_of_unknowns[i] - previous_x[i])
            denominator = abs(values_of_unknowns[i])

            if denominator != 0:
                relative_error = delta / denominator
                if relative_error > max_relative_error:
                    max_relative_error = relative_error
            # Handle case where the true value is zero (use absolute error)
            elif delta != 0:
                # If the value is 0 but changed, we haven't converged
                max_relative_error = Decimal("1")
        addsteps(all_steps,f"Iteration {k+1}: Relative Error = {max_relative_error:.{item.precision}f}",item.matrix,values_of_unknowns)
        if max_relative_error < item.tolerance:
            # Convergence achieved
            timer_stop = time.perf_counter()
            Response("SUCCESS",values_of_unknowns,round(timer_stop - timer_start,6),0,all_steps,"")
        timer_stop=time.perf_counter()
    return Response("Error",values_of_unknowns,round(timer_stop-timer_start,6),item.max_iterations,all_steps,f"error: Did not converge within {item.max_iterations} iterations. Final error: {max_relative_error}") 

# ! Jacobi Method (without pivoting)
def Jacobi_method(item: Item,all_steps:List['Steps']):
    timer_start = time.perf_counter()
    getcontext().prec = item.precision if item.precision is not None else 10
    values_of_unknowns = item.initial_guess[:]
    # Check for zero pivots (diagonal elements) before starting
    for i in range(item.size):
        if item.matrix[i][i] == 0:
            timer_stop = time.perf_counter()
            addsteps(all_steps,"Cant use Gauss Seidel because the pivot is zero and we cant divide by zero",item.matrix,item.vector_of_sol)
            return Response("Error",item.vector_of_sol,round(timer_stop - timer_start,6),0,all_steps,"can't divide by zero")

    for k in range(item.max_iterations):
        previous_x = values_of_unknowns[:]
        for row in range(item.size):
            s = Decimal("0")  # s == sum
            for col in range(item.size):
                if col != row:
                    s += previous_x[col] * item.matrix[row][col]
            values_of_unknowns[row] = (
                item.vector_of_sol[row] - s) / item.matrix[row][row]
        # --- Check for Convergence ---
        # Compute the maximum relative change between the current and previous vector
        max_relative_error = Decimal("0")
        for i in range(item.size):
            # |current - previous| / |current|
            delta = abs(values_of_unknowns[i] - previous_x[i])
            denominator = abs(values_of_unknowns[i])

            if denominator != 0:
                relative_error = delta / denominator
                if relative_error > max_relative_error:
                    max_relative_error = relative_error
            # Handle case where the true value is zero (use absolute error)
            elif delta != 0:
                # If the value is 0 but changed, we haven't converged
                max_relative_error = Decimal("1")
        if max_relative_error < item.Tolerance:
            # Convergence achieved
            timer_stop = time.perf_counter()
            return Response("SUCCESS",values_of_unknowns,round(timer_stop - timer_start,6),k+1,all_steps,"")
        timer_stop = time.perf_counter()
    return Response("Error",values_of_unknowns,round(timer_stop-timer_start,6),item.max_iterations,all_steps,f"error: Did not converge within {item.max_iterations} iterations. Final error: {max_relative_error}")

# ! LU Decomposition Method (Doolittle's Method)
def LU_decomposition_Doolittle_method(item: Item,all_steps:List['Steps']):
    getcontext().prec = item.precision if item.precision is not None else 10
    # Step 1: LU Decomposition (Doolittle's Method)
    L = [[Decimal("0") for _ in range(item.size)] for _ in range(item.size)]
    U = [[Decimal("0") for _ in range(item.size)] for _ in range(item.size)]

    for i in range(item.size):
        # Upper Triangular
        for j in range(i, item.size):
            sum_u = Decimal("0")
            for k in range(i):
                sum_u += L[i][k] * U[k][j]
            U[i][j] = item.matrix[i][j] - sum_u
        # Lower Triangular
        for j in range(i, item.size):
            if i == j:
                L[i][i] = Decimal("1")  # Diagonal as 1
            else:
                sum_l = Decimal("0")
                for k in range(i):
                    sum_l += L[j][k] * U[k][i]
                if U[i][i] == 0:
                    return "error"  # Singular matrix
                L[j][i] = (item.matrix[j][i] - sum_l) / U[i][i]

    # Step 2: Solve Ly = b using forward substitution
    y = [Decimal("0") for _ in range(item.size)]
    for i in range(item.size):
        sum_y = Decimal("0")
        for j in range(i):
            sum_y += L[i][j] * y[j]
        y[i] = item.vector_of_sol[i] - sum_y
    # Step 3: Solve Ux = y using backward substitution
    x = [Decimal("0") for _ in range(item.size)]
    for i in range(item.size-1, -1, -1):
        sum_x = Decimal("0")
        for j in range(i+1, item.size):
            sum_x += U[i][j] * x[j]
        if U[i][i] == 0:
            return "error"  # Singular matrix
        x[i] = (y[i] - sum_x) / U[i][i]

    return x

# ! LU Decomposition Method (Crout's Method)
def LU_decomposition_Crout_method(item: Item,all_steps:List['Steps']):
    getcontext().prec = item.precision if item.precision is not None else 10
    # Step 1: LU Decomposition (Crout's Method)
    L = [[Decimal("0") for _ in range(item.size)] for _ in range(item.size)]
    U = [[Decimal("0") for _ in range(item.size)] for _ in range(item.size)]

    for i in range(item.size):
        # Lower Triangular
        for j in range(i, item.size):
            sum_l = Decimal("0")
            for k in range(i):
                sum_l += L[j][k] * U[k][i]
            L[j][i] = item.matrix[j][i] - sum_l
        # Upper Triangular
        for j in range(i, item.size):
            if i == j:
                U[i][i] = Decimal("1")  # Diagonal as 1
            else:
                sum_u = Decimal("0")
                for k in range(i):
                    sum_u += L[i][k] * U[k][j]
                if L[i][i] == 0:
                    return "error"  # Singular matrix
                U[i][j] = (item.matrix[i][j] - sum_u) / L[i][i]

    # Step 2: Solve Ly = b using forward substitution
    y = [Decimal("0") for _ in range(item.size)]
    for i in range(item.size):
        sum_y = Decimal("0")
        for j in range(i):
            sum_y += L[i][j] * y[j]
        y[i] = item.vector_of_sol[i] - sum_y
    # Step 3: Solve Ux = y using backward substitution
    x = [Decimal("0") for _ in range(item.size)]
    for i in range(item.size-1, -1, -1):
        sum_x = Decimal("0")
        for j in range(i+1, item.size):
            sum_x += U[i][j] * x[j]
        if U[i][i] == 0:
            return "error"  # Singular matrix
        x[i] = (y[i] - sum_x) / U[i][i]

    return x

# ! LU Decomposition Method (Cholesky's Method)
def LU_decomposition_Cholesky_method(item: Item,all_steps:List['Steps']):
    """
    Performs Cholesky Decomposition (A = L * L^T) to solve Ax = b.
    Requires matrix A to be symmetric and positive-definite.
    L is a lower triangular matrix.
    """

    L = [[Decimal("0") for _ in range(item.size)] for _ in range(item.size)]

    # --- Step 1: Cholesky Decomposition (A = L * L^T) ---

    for i in range(item.size):
        for j in range(i + 1):  # Only compute elements in the lower triangle

            # Calculate the sum: sum(L[i][k] * L[j][k])
            sum_val = Decimal("0")
            for k in range(j):
                sum_val += L[i][k] * L[j][k]

            if i == j:  # Diagonal elements: L[i][i]
                # Formula: L[i][i] = sqrt(A[i][i] - sum(L[i][k]^2))

                # Check for negative value under the square root (not positive-definite)
                # We use the square root function from the math module, but must convert to/from Decimal
                value = item.matrix[i][i] - sum_val
                if value <= 0:
                    # Matrix is not positive definite
                    return "error: Matrix is not positive-definite (Cholesky failed)"

                # To maintain high precision with Decimal, we must import math.sqrt
                # Since we don't know if math.sqrt is imported, we'll assume a Decimal-compatible sqrt is available
                # or handle the conversion (best to import `decimal.Decimal` and `math` at the top of the file)
                try:
                    # Decimal objects have a .sqrt() method
                    L[i][i] = value.sqrt()
                except AttributeError:
                    # Fallback if Decimal().sqrt() is not available (though it should be with Decimal)
                    return "error: Missing Decimal sqrt method"

            else:  # Off-diagonal elements: L[i][j] where i > j
                # Formula: L[i][j] = (A[i][j] - sum(L[i][k] * L[j][k])) / L[j][j]

                if L[j][j] == 0:
                    return "error: Zero diagonal element encountered"

                L[i][j] = (item.matrix[i][j] - sum_val) / L[j][j]

    # --- Step 2: Solve Ly = b using Forward Substitution ---
    # L is the lower triangular matrix

    y = [Decimal("0") for _ in range(item.size)]
    for i in range(item.size):
        sum_y = Decimal("0")
        for j in range(i):
            sum_y += L[i][j] * y[j]

        # Since L[i][i] is calculated and stored, we must divide by it here
        # (L[i][i] should never be zero for positive-definite matrices)
        if L[i][i] == 0:
            return "error: L[i][i] zero during forward substitution"

        y[i] = (item.vector_of_sol[i] - sum_y) / L[i][i]

    # --- Step 3: Solve L^T x = y using Backward Substitution ---
    # L^T is the upper triangular matrix

    x = [Decimal("0") for _ in range(item.size)]
    for i in range(item.size - 1, -1, -1):
        sum_x = Decimal("0")
        for j in range(i + 1, item.size):
            # L^T[i][j] is L[j][i]
            sum_x += L[j][i] * x[j]

        # The pivot L^T[i][i] is L[i][i]
        if L[i][i] == 0:
            return "error: L[i][i] zero during backward substitution"

        x[i] = (y[i] - sum_x) / L[i][i]

    return x