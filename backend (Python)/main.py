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
    max_iterations:int | None = 100
    Tolerance : Decimal | None = Decimal("1e-5")
    methodParams: dict | None = {}
    

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
    status = helper.check_havesol(item.size,B_copy,A_copy,all_steps)
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
    status = helper.check_havesol(item.size,B_copy,A_copy,all_steps)
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
    if helper.forward_elimination_withPivoting_and_scaling(item.size, A_copy, B_copy,all_steps) == "error":
       end_time = time.perf_counter()
       return Response("Error",[],round(end_time-start_time,6),0,all_steps,"Singluar matrix")
    status = helper.check_havesol(item.size,B_copy,A_copy,all_steps)
    if(status != "unique"):
        end_time = time.perf_counter()
        if(status == "None"):
            return Response("Error",[],round(end_time-start_time,6),0,all_steps,"the system has no solution")
        else:
            return Response("Error",[],round(end_time-start_time,6),0,all_steps,"the system has Infinite number of solution")
    vector_of_unknowns = helper.backward_substitution(item.size, A_copy, B_copy,all_steps)
    end_time = time.perf_counter()
    return Response("SUCCESS",vector_of_unknowns,round(end_time - start_time,6),0,all_steps,"")

# ! Gauss-Jordan elimination (with partial pivoting)
def Gauss_Jordan_elimination(item: Item,all_steps:List['Steps']):
    start_time = time.perf_counter()
    getcontext().prec = item.precision if item.precision is not None else 10
    error_status = helper.forward_elimination_withPivoting(item.size,item.matrix,item.vector_of_sol,all_steps)
    if error_status == "error":
        end_time = time.perf_counter()
        return Response("Error",[],round(end_time-start_time,6),0,all_steps,"Singluar matrix")
    status = helper.check_havesol(item.size, item.vector_of_sol, item.matrix, all_steps)
    if (status != "unique"):
        end_time = time.perf_counter()
        if (status == "None"):
            return Response("Error", [], round(end_time - start_time, 6), 0, all_steps, "the system has no solution")
        else:
            return Response("Error", [], round(end_time - start_time, 6), 0, all_steps,"the system has Infinite number of solution")
    helper.normalize_matrix(item.size,item.vector_of_sol,item.matrix,all_steps)
    helper.backward_elimination(item.size,item.vector_of_sol,item.matrix, all_steps)
    end_time = time.perf_counter()
    return Response("SUCCESS",item.vector_of_sol,round(end_time-start_time,6),0,all_steps,"")

# ! Gauss-Seidel Method (without pivoting)
def Gauss_Seidel_method(item: Item,all_steps:List['Steps']):
    iterations=0
    equa = []

    timer_start = time.perf_counter()
    getcontext().prec = item.precision if item.precision is not None else 10
    values_of_unknowns = [Decimal(str(x)) for x in item.initial_guess]
    # Check for zero pivots (diagonal elements) before starting
    for i in range(item.size):
        if item.matrix[i][i] == 0:
            timer_stop = time.perf_counter()
            addsteps(all_steps,"Cant use Gauss Seidel because the pivot is zero and we cant divide by zero",item.matrix,item.vector_of_sol)
            return Response("Error",item.vector_of_sol,round(timer_stop - timer_start,6),0,all_steps,"can't divide by zero")
    diagonal = helper.check_diagonally_dominant(item.size,item.vector_of_sol,item.matrix,all_steps)
    for i in range(item.size):
        current = f'X{i+1} = ({item.vector_of_sol[i]}'
        for j in range(item.size):
            if i!=j:
                value = item.matrix[i][j]
                status = "(old)"
                if j<i:
                    status = "(new)"
                current += f"-{value}X{j+1}<sup>{status}</sup>"
        current += f" ) / {item.matrix[i][i]}"

        equa.append(current)
    for k in range(item.max_iterations):
        iterations+=1
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
        vec_str = ", ".join([f"{x:.{item.precision}f}" for x in values_of_unknowns])

                # Combine Vector + Error into one message
        addsteps(all_steps,f"Iter {k + 1}: Vector=[{vec_str}] | Error={max_relative_error:.{item.precision}f}",item.matrix,values_of_unknowns,Error=max_relative_error)
        if max_relative_error < item.Tolerance:
            # Convergence achieved
            timer_stop = time.perf_counter()
            return Response("SUCCESS",values_of_unknowns,round(timer_stop - timer_start,6),iterations,all_steps,"",equations=equa,Diagonal=diagonal)
    timer_stop=time.perf_counter()
    return Response("Error",values_of_unknowns,round(timer_stop-timer_start,6),item.max_iterations,all_steps,f"error: Did not converge within {item.max_iterations} iterations. Final error: {max_relative_error}",equations=equa,Diagonal=diagonal)

# ! Jacobi Method (without pivoting)
def Jacobi_method(item: Item,all_steps:List['Steps']):
    timer_start = time.perf_counter()
    iteration=0
    equa=[]
    getcontext().prec = item.precision if item.precision is not None else 10
    values_of_unknowns = item.initial_guess[:]
    # Check for zero pivots (diagonal elements) before starting
    for i in range(item.size):
        if item.matrix[i][i] == 0:
            timer_stop = time.perf_counter()
            addsteps(all_steps,"Cant use Gauss Seidel because the pivot is zero and we cant divide by zero",item.matrix,item.vector_of_sol)
            return Response("Error",item.vector_of_sol,round(timer_stop - timer_start,6),0,all_steps,"can't divide by zero")
    for i in range(item.size):
        current = f'X{i+1} = ({item.vector_of_sol[i]}'
        for j in range(item.size):
            if i!=j:
                value = item.matrix[i][j]
                status = "(old)"
                current += f"-{value}X{j+1}<sup>{status}</sup>"
        current += f" ) / {item.matrix[i][i]}"

        equa.append(current)
    for k in range(item.max_iterations):
        iteration +=1
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
        vec_str = ", ".join([f"{x:.{item.precision}f}" for x in values_of_unknowns])

        # Combine Vector + Error into one message
        addsteps(all_steps, f"Iter {k + 1}: Vector=[{vec_str}] | Error={max_relative_error:.{item.precision}f}",
                 item.matrix, values_of_unknowns,Error=max_relative_error)
        if max_relative_error < item.Tolerance:
            # Convergence achieved
            timer_stop = time.perf_counter()
            return Response("SUCCESS",values_of_unknowns,round(timer_stop - timer_start,6),iteration,all_steps,"")
    timer_stop = time.perf_counter()
    return Response("Error",values_of_unknowns,round(timer_stop-timer_start,6),item.max_iterations,all_steps,f"error: Did not converge within {item.max_iterations} iterations. Final error: {max_relative_error}")

# ! LU Decomposition Method (Doolittle's Method)
def LU_decomposition_Doolittle_method(item: Item,all_steps:List['Steps']):
    getcontext().prec = item.precision if item.precision is not None else 10
    timer_start = time.perf_counter()
    # Step 1: LU Decomposition (Doolittle's Method)
    U = [[Decimal("0") for _ in range(item.size)] for _ in range(item.size)]
    L = [[ Decimal("0") for _ in range(item.size)] for _ in range(item.size) ]
    for i in range(item.size):
        for j in range(item.size):
            if(i==j):
                L[i][j] = Decimal("1")

    for i in range(item.size):
        # Upper Triangular
        for j in range(i, item.size):
            sum_u = Decimal("0")
            for k in range(i):
                sum_u += L[i][k] * U[k][j]
            U[i][j] = item.matrix[i][j] - sum_u
            addsteps(all_steps,f"U{i}{j} = A{i}{j} - {sum_u}",U,item.vector_of_sol,L,U)
        # Lower Triangular
        for j in range(i, item.size):
            if i == j:
                L[i][i] = Decimal("1")  # Diagonal as 1
            else:
                sum_l = Decimal("0")
                for k in range(i):
                    sum_l += L[j][k] * U[k][i]
                if U[i][i] == 0:
                    timer_end = time.perf_counter()
                    return Response("ERROR",item.vector_of_sol,round(timer_end-timer_start,6),0,all_steps,"Singular matrix")  # Singular matrix
                L[j][i] = (item.matrix[j][i] - sum_l) / U[i][i]
                addsteps(all_steps,f"L{j}{i} = (A{j}{i} - {sum_l})/U{i}{i}",L,item.vector_of_sol,L,U)

    # Step 2: Solve Ly = b using forward substitution
    addsteps(all_steps,"solving Ly=b using forward substitution",L,item.vector_of_sol,L,U)
    y = [Decimal("0") for _ in range(item.size)]
    for i in range(item.size):
        sum_y = Decimal("0")
        for j in range(i):
            sum_y += L[i][j] * y[j]
        y[i] = item.vector_of_sol[i] - sum_y
        addsteps(all_steps,f"y{i} = b{i} - {sum_y}",L,y,L,U)
    # Step 3: Solve Ux = y using backward substitution
    addsteps(all_steps,"solving Ux=y using backward substitution",U,y,L,U)
    x = [Decimal("0") for _ in range(item.size)]
    for i in range(item.size-1, -1, -1):
        sum_x = Decimal("0")
        for j in range(i+1, item.size):
            sum_x += U[i][j] * x[j]
        if U[i][i] == 0:
            timer_end = time.perf_counter()
            return Response("ERROR",item.vector_of_sol,round(timer_end-timer_start,6),0,all_steps,"Singular matrix")
        x[i] = (y[i] - sum_x) / U[i][i]
        addsteps(all_steps,f"x{i} = (y{i} - {sum_x})/ U{i}{i}",U,x,L,U)
    timer_stop = time.perf_counter()
    return Response("SUCCESS",x,round(timer_stop-timer_start,6),item.max_iterations,all_steps,"",L,U)

# ! LU Decomposition Method (Crout's Method)
def LU_decomposition_Crout_method(item: Item,all_steps:List['Steps']):
    timer_start = time.perf_counter()
    getcontext().prec = item.precision if item.precision is not None else 10
    # Step 1: LU Decomposition (Crout's Method)
    L = [[Decimal("0") for _ in range(item.size)] for _ in range(item.size)]
    U = [[Decimal("0") for _ in range(item.size)] for _ in range(item.size)]
    for i in range(item.size):
        for j in range(item.size):
            if(i==j):
                U[i][j] = Decimal("1")
    for i in range(item.size):
        # Lower Triangular
        for j in range(i, item.size):
            sum_l = Decimal("0")
            for k in range(i):
                sum_l += L[j][k] * U[k][i]
            L[j][i] = item.matrix[j][i] - sum_l
            addsteps(all_steps,f"L{j}{i} = A{j}{i} - {sum_l}",L,item.vector_of_sol,L,U)
        # Upper Triangular
        for j in range(i, item.size):
            if i == j:
                U[i][i] = Decimal("1")  # Diagonal as 1
            else:
                sum_u = Decimal("0")
                for k in range(i):
                    sum_u += L[i][k] * U[k][j]
                if L[i][i] == 0:
                    timer_end = time.perf_counter()
                    return Response("ERROR",item.vector_of_sol,round(timer_end-timer_start,6),0,all_steps,"Singular matrix") # Singular matrix
                U[i][j] = (item.matrix[i][j] - sum_u) / L[i][i]
                addsteps(all_steps, f"U{i}{j} = (A{i}{j} - {sum_u})/L{i}{i}", U, item.vector_of_sol, L, U)

    # Step 2: Solve Ly = b using forward substitution
    addsteps(all_steps,"solving Ly=b using forward substitution",L,item.vector_of_sol,L,U)
    y = [Decimal("0") for _ in range(item.size)]
    for i in range(item.size):
        sum_y = Decimal("0")
        for j in range(i):
            sum_y += L[i][j] * y[j]
        if L[i][i] == 0:
            timer_end = time.perf_counter()
            return Response("ERROR", item.vector_of_sol, round(timer_end - timer_start, 6), 0, all_steps,
                            "Singular matrix (Pivot is zero)")
        y[i] = (item.vector_of_sol[i] - sum_y)/L[i][i]
        addsteps(all_steps,f"y{i} = (b{i}-{sum_y})/L{i}{i}",L,y,L,U)
    # Step 3: Solve Ux = y using backward substitution
    addsteps(all_steps, "solving Ux=y using forward substitution", U, item.vector_of_sol, L, U)
    x = [Decimal("0") for _ in range(item.size)]
    for i in range(item.size-1, -1, -1):
        sum_x = Decimal("0")
        for j in range(i+1, item.size):
            sum_x += U[i][j] * x[j]
        if U[i][i] == 0:
            timer_end = time.perf_counter()
            return Response("ERROR", item.vector_of_sol, round(timer_end - timer_start, 6), 0, all_steps,
                            "Singular matrix")  # Singular matrix
        x[i] = (y[i] - sum_x) / U[i][i]
        addsteps(all_steps, f"x{i} = (b{i}-{sum_x})/U{i}{i}", U, x, L, U)
    timer_stop = time.perf_counter()
    return Response("SUCCESS",x,round(timer_stop-timer_start,6),item.max_iterations,all_steps,"",L,U)

# ! LU Decomposition Method (Cholesky's Method)
def LU_decomposition_Cholesky_method(item: Item,all_steps:List['Steps']):
    """
    Performs Cholesky Decomposition (A = L * L^T) to solve Ax = b.
    Requires matrix A to be symmetric and positive-definite.
    L is a lower triangular matrix.
    """
    timer_start = time.perf_counter()

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
                    timer_end = time.perf_counter()
                    addsteps(all_steps,f"check for positive defines A({i},{i}) - {sum_val} > 0 ?",item.matrix,item.vector_of_sol)
                    return Response("ERROR",item.vector_of_sol,round(timer_end - timer_start,6),0,all_steps,"Matrix not positive definite")

                # To maintain high precision with Decimal, we must import math.sqrt
                # Since we don't know if math.sqrt is imported, we'll assume a Decimal-compatible sqrt is available
                # or handle the conversion (best to import `decimal.Decimal` and `math` at the top of the file)
                try:
                    # Decimal objects have a .sqrt() method
                    L[i][i] = value.sqrt()
                    addsteps(all_steps,f"L({i},{i} = √(A({i},{i} - {sum_val})))",L,item.vector_of_sol,L)
                except AttributeError:
                    # Fallback if Decimal().sqrt() is not available (though it should be with Decimal)
                    return "error: Missing Decimal sqrt method"

            else:  # Off-diagonal elements: L[i][j] where i > j
                # Formula: L[i][j] = (A[i][j] - sum(L[i][k] * L[j][k])) / L[j][j]

                if L[j][j] == 0:
                    timer_end = time.perf_counter()
                    return Response("ERROR",item.vector_of_sol,round(timer_end-timer_start,6),0,all_steps,"Zero diagonal element encountered")

                L[i][j] = (item.matrix[i][j] - sum_val) / L[j][j]
                addsteps(all_steps,f"L({i},{j}) = (A({i},{j}) - {sum_val})/L({j},{j})",L,item.vector_of_sol,L)

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
            timer_end = time.perf_counter()
            addsteps(all_steps,"check for diagonals = zero",L,item.vector_of_sol,L)
            return Response("ERROR",item.vector_of_sol,round(timer_end-timer_start,6),0,all_steps,f"L{i},{i} zero during forward substitution")

        y[i] = (item.vector_of_sol[i] - sum_y) / L[i][i]
        addsteps(all_steps,f"y({i}) = (b({i}) - {sum_y})/L({i},{i})",L,y,L)

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
            timer_end = time.perf_counter()
            addsteps(all_steps, "check for diagonals = zero", L, item.vector_of_sol, L)
            return Response("ERROR",item.vector_of_sol,round(timer_end-timer_start,6),0,all_steps,f"L{i},{i} zero during forward substitution")

        x[i] = (y[i] - sum_x) / L[i][i]
        addsteps(all_steps, f"x({i}) = (y({i}) - {sum_x})/L({i},{i})", L, x, L)
    timer_end = time.perf_counter()
    return Response("SUCCESS",x,round(timer_end-timer_start,6),0,all_steps,"",L)