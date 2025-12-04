from decimal import Decimal, getcontext
from pydantic import BaseModel
import time
from response import Response,Steps,addsteps
import helper
from typing_extensions import List
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
    diagonl = helper.check_diagonally_dominant(item.size,values_of_unknowns,item.matrix,all_steps)
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
            return Response("SUCCESS",values_of_unknowns,round(timer_stop - timer_start,6),iteration,all_steps,"",Diagonal=diagonl,equations=equa)
    timer_stop = time.perf_counter()
    return Response("Error",values_of_unknowns,round(timer_stop-timer_start,6),item.max_iterations,all_steps,f"error: Did not converge within {item.max_iterations} iterations. Final error: {max_relative_error}",Diagonal=diagonl,equations=equa)

def LU_decomposition_Doolittle_method(item: Item, all_steps: List["Steps"]):
    getcontext().prec = item.precision if item.precision is not None else 10
    timer_start = time.perf_counter()

    n = item.size

    # Initialize L as identity and U as a copy of A
    L = [[Decimal("0") for _ in range(n)] for _ in range(n)]
    U = [[item.matrix[i][j] for j in range(n)] for i in range(n)]

    for i in range(n):
        L[i][i] = Decimal("1")  # Diagonal entries = 1

    # ---------------------------
    #  Step 1: Forward Elimination
    # ---------------------------
    flag = False
    for i in range(n - 1):  # pivot row
        if U[i][i] == 0:
            for j in range(i+1,n-1):
                if U[j][i] > 0 and j!= n-1:
                    Swap = U[i]
                    U[i]=U[j]
                    U[j]= Swap
                    flag =True
                    break
            if not flag:
                timer_end = time.perf_counter()
                return Response("ERROR", item.vector_of_sol, round(timer_end - timer_start, 6),
                                0, all_steps, "Singular matrix")

        for j in range(i + 1, n):  # eliminate rows below
            multiplier = U[j][i] / U[i][i]
            L[j][i] = multiplier
            addsteps(all_steps,
                     f"L{j+1}{i+1} = A{j+1}{i+1} / A{i+1}{i+1} = {multiplier}",
                     L, item.vector_of_sol, L, U,None,pivotIndex={"r":i,"c":i},highlightRow=j)

            # Update row j in U
            for k in range(i, n):
                old_value = U[j][k]
                U[j][k] = U[j][k] - multiplier * U[i][k]

            addsteps(all_steps,
                     f"U{j+1} = U{j+1} - ({multiplier}) * U{i+1}",
                     U, item.vector_of_sol, L, U,pivotIndex={"r":i,"c":i},highlightRow=j)

    # ---------------------------
    #  Step 2: Forward Substitution (Ly = b)
    # ---------------------------
    addsteps(all_steps, "Solving Ly=b using forward substitution", L,
             item.vector_of_sol, L, U)

    y = [Decimal("0") for _ in range(n)]
    for i in range(n):
        sum_y = sum(L[i][j] * y[j] for j in range(i))
        y[i] = item.vector_of_sol[i] - sum_y
        addsteps(all_steps,
                 f"y{i+1} = b{i+1} - {sum_y}",
                 L, y, L, U,pivotIndex={"r":i,"c":i},highlightRow=i)

    # ---------------------------
    #  Step 3: Backward Substitution (Ux = y)
    # ---------------------------
    addsteps(all_steps, "Solving Ux=y using backward substitution", U, y, L, U)

    x = [Decimal("0") for _ in range(n)]
    for i in range(n - 1, -1, -1):
        if U[i][i] == 0:
            timer_end = time.perf_counter()
            return Response("ERROR", item.vector_of_sol, round(timer_end - timer_start, 6),
                            0, all_steps, "Singular matrix")

        sum_x = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - sum_x) / U[i][i]

        addsteps(all_steps,
                 f"X{i+1} = (y{i+1} - {sum_x}) / U{i+1}{i+1}",
                 U, x, L, U,pivotIndex={"r":i,"c":i},highlightRow=i)

    timer_stop = time.perf_counter()
    return Response("SUCCESS", x, round(timer_stop - timer_start, 6),
                    item.max_iterations, all_steps,"",L,U)

# ! LU Decomposition Method (Crout's Method)
# def LU_decomposition_Crout_method(item: Item,all_steps:List['Steps']):
#     timer_start = time.perf_counter()
#     getcontext().prec = item.precision if item.precision is not None else 10
#     # Step 1: LU Decomposition (Crout's Method)
#     L = [[Decimal("0") for _ in range(item.size)] for _ in range(item.size)]
#     U = [[Decimal("0") for _ in range(item.size)] for _ in range(item.size)]
#     for i in range(item.size):
#         for j in range(item.size):
#             if(i==j):
#                 U[i][j] = Decimal("1")
#     for i in range(item.size):
#         # Lower Triangular
#         for j in range(i, item.size):
#             sum_l = Decimal("0")
#             for k in range(i):
#                 sum_l += L[j][k] * U[k][i]
#             L[j][i] = item.matrix[j][i] - sum_l
#             addsteps(all_steps,f"L{j}{i} = A{j}{i} - {sum_l}",L,item.vector_of_sol,L,U)
#         # Upper Triangular
#         for j in range(i, item.size):
#             if i == j:
#                 U[i][i] = Decimal("1")  # Diagonal as 1
#             else:
#                 sum_u = Decimal("0")
#                 for k in range(i):
#                     sum_u += L[i][k] * U[k][j]
#                 if L[i][i] == 0:
#                     timer_end = time.perf_counter()
#                     return Response("ERROR",item.vector_of_sol,round(timer_end-timer_start,6),0,all_steps,"Singular matrix") # Singular matrix
#                 U[i][j] = (item.matrix[i][j] - sum_u) / L[i][i]
#                 addsteps(all_steps, f"U{i}{j} = (A{i}{j} - {sum_u})/L{i}{i}", U, item.vector_of_sol, L, U)

#     # Step 2: Solve Ly = b using forward substitution
#     addsteps(all_steps,"solving Ly=b using forward substitution",L,item.vector_of_sol,L,U)
#     y = [Decimal("0") for _ in range(item.size)]
#     for i in range(item.size):
#         sum_y = Decimal("0")
#         for j in range(i):
#             sum_y += L[i][j] * y[j]
#         if L[i][i] == 0:
#             timer_end = time.perf_counter()
#             return Response("ERROR", item.vector_of_sol, round(timer_end - timer_start, 6), 0, all_steps,
#                             "Singular matrix (Pivot is zero)")
#         y[i] = (item.vector_of_sol[i] - sum_y)/L[i][i]
#         addsteps(all_steps,f"y{i} = (b{i}-{sum_y})/L{i}{i}",L,y,L,U)
#     # Step 3: Solve Ux = y using backward substitution
#     addsteps(all_steps, "solving Ux=y using forward substitution", U, item.vector_of_sol, L, U)
#     x = [Decimal("0") for _ in range(item.size)]
#     for i in range(item.size-1, -1, -1):
#         sum_x = Decimal("0")
#         for j in range(i+1, item.size):
#             sum_x += U[i][j] * x[j]
#         if U[i][i] == 0:
#             timer_end = time.perf_counter()
#             return Response("ERROR", item.vector_of_sol, round(timer_end - timer_start, 6), 0, all_steps,
#                             "Singular matrix")  # Singular matrix
#         x[i] = (y[i] - sum_x) / U[i][i]
#         addsteps(all_steps, f"x{i} = (b{i}-{sum_x})/U{i}{i}", U, x, L, U)
#     timer_stop = time.perf_counter()
#     return Response("SUCCESS",x,round(timer_stop-timer_start,6),item.max_iterations,all_steps,"",L,U)
def LU_decomposition_Crout_method(item: Item, all_steps: List['Steps']):
    timer_start = time.perf_counter()
    getcontext().prec = item.precision if item.precision is not None else 10

    n = item.size

    # Initialize L and U
    L = [[Decimal("0") for _ in range(n)] for _ in range(n)]
    U = [[Decimal("0") for _ in range(n)] for _ in range(n)]

    # U diagonal = 1
    for i in range(n):
        U[i][i] = Decimal("1")

    A = item.matrix

    # --------------------------
    # Step 1: First column of L
    # --------------------------
    for i in range(n):
        L[i][0] = A[i][0]
        addsteps(all_steps, f"L[{i+1}][1] = A[{i+1}][1]", L, item.vector_of_sol, L, U,pivotIndex={"r":i,"c":0})

    # --------------------------
    # Step 2: First row of U
    # --------------------------
    if L[0][0] == 0:
        return Response("ERROR", item.vector_of_sol, 0, 0, all_steps, "Singular matrix")

    for j in range(1, n):
        U[0][j] = A[0][j] / L[0][0]
        addsteps(all_steps, f"U[1][{j+1}] = A[1][{j+1}] / L[1][1]", U, item.vector_of_sol, L, U,pivotIndex={"r":0,"c":j})

    # --------------------------
    # Step 3: For columns j = 2..n-1
    # --------------------------
    for j in range(1, n - 1):

        # ---- Compute L[i][j] for i>=j ----
        for i in range(j, n):
            sum_l = Decimal("0")
            for k in range(j):
                sum_l += L[i][k] * U[k][j]

            L[i][j] = A[i][j] - sum_l
            addsteps(all_steps, f"L[{i+1}][{j+1}] = A[{i+1}][{j+1}] - {sum_l}", L, item.vector_of_sol, L, U,pivotIndex={"r":i,"c":j})

        # ---- Compute U[j][k] for k=j+1..n ----
        if L[j][j] == 0:
            return Response("ERROR", item.vector_of_sol, 0, 0, all_steps,
                            "Singular matrix (Zero pivot)")

        for k in range(j + 1, n):
            sum_u = Decimal("0")
            for i2 in range(j):
                sum_u += L[j][i2] * U[i2][k]

            U[j][k] = (A[j][k] - sum_u) / L[j][j]
            addsteps(all_steps,
                     f"U[{j+1}][{k+1}] = (A[{j+1}][{k+1}] - {sum_u}) / L[{j+1}][{j+1}]",
                     U, item.vector_of_sol, L, U,pivotIndex={"r":j,"c":k})

    # --------------------------
    # Step 4: Compute the last L[n-1][n-1]
    # --------------------------
    sum_last = Decimal("0")
    for k in range(n - 1):
        sum_last += L[n - 1][k] * U[k][n - 1]

    L[n - 1][n - 1] = A[n - 1][n - 1] - sum_last
    addsteps(all_steps,
             f"L[{n}][{n}] = A[{n}][{n}] - {sum_last}",
             L, item.vector_of_sol, L, U,pivotIndex={"r":n-1,"c":n-1})

    # --------------------------
    # Step 5: Forward substitution Ly = b
    # --------------------------
    y = [Decimal("0") for _ in range(n)]

    for i in range(n):
        s = Decimal("0")
        for j in range(i):
            s += L[i][j] * y[j]

        if L[i][i] == 0:
            return Response("ERROR", item.vector_of_sol, 0, 0, all_steps, "Singular matrix")

        y[i] = (item.vector_of_sol[i] - s) / L[i][i]
        addsteps(all_steps, f"y[{i+1}] = (b[{i+1}] - {s}) / L[{i+1}][{i+1}]", L, y, L, U,pivotIndex={"r":i,"c":i})

    # --------------------------
    # Step 6: Backward substitution Ux = y
    # --------------------------
    x = [Decimal("0") for _ in range(n)]

    for i in range(n - 1, -1, -1):
        s = Decimal("0")
        for j in range(i + 1, n):
            s += U[i][j] * x[j]

        x[i] = y[i] - s  # since U[i][i] = 1
        addsteps(all_steps, f"x[{i+1}] = y[{i+1}] - {s}", U, x, L, U,pivotIndex={"r":i,"c":i})

    timer_end = time.perf_counter()
    return Response("SUCCESS", x, round(timer_end - timer_start, 6),
                    item.max_iterations, all_steps, "", L, U)


# ! LU Decomposition Method (Cholesky's Method)
# def LU_decomposition_Cholesky_method(item: Item,all_steps:List['Steps']):
#     """
#     Performs Cholesky Decomposition (A = L * L^T) to solve Ax = b.
#     Requires matrix A to be symmetric and positive-definite.
#     L is a lower triangular matrix.
#     """
#     timer_start = time.perf_counter()

#     L = [[Decimal("0") for _ in range(item.size)] for _ in range(item.size)]

#     # --- Step 1: Cholesky Decomposition (A = L * L^T) ---

#     for i in range(item.size):
#         for j in range(i + 1):  # Only compute elements in the lower triangle

#             # Calculate the sum: sum(L[i][k] * L[j][k])
#             sum_val = Decimal("0")
#             for k in range(j):
#                 sum_val += L[i][k] * L[j][k]

#             if i == j:  # Diagonal elements: L[i][i]
#                 # Formula: L[i][i] = sqrt(A[i][i] - sum(L[i][k]^2))

#                 # Check for negative value under the square root (not positive-definite)
#                 # We use the square root function from the math module, but must convert to/from Decimal
#                 value = item.matrix[i][i] - sum_val
#                 if value <= 0:
#                     # Matrix is not positive definite
#                     timer_end = time.perf_counter()
#                     addsteps(all_steps,f"check for positive defines A({i},{i}) - {sum_val} > 0 ?",item.matrix,item.vector_of_sol)
#                     return Response("ERROR",item.vector_of_sol,round(timer_end - timer_start,6),0,all_steps,"Matrix not positive definite")

#                 # To maintain high precision with Decimal, we must import math.sqrt
#                 # Since we don't know if math.sqrt is imported, we'll assume a Decimal-compatible sqrt is available
#                 # or handle the conversion (best to import `decimal.Decimal` and `math` at the top of the file)
#                 try:
#                     # Decimal objects have a .sqrt() method
#                     L[i][i] = value.sqrt()
#                     addsteps(all_steps,f"L({i},{i} = √(A({i},{i} - {sum_val})))",L,item.vector_of_sol,L)
#                 except AttributeError:
#                     # Fallback if Decimal().sqrt() is not available (though it should be with Decimal)
#                     return "error: Missing Decimal sqrt method"

#             else:  # Off-diagonal elements: L[i][j] where i > j
#                 # Formula: L[i][j] = (A[i][j] - sum(L[i][k] * L[j][k])) / L[j][j]

#                 if L[j][j] == 0:
#                     timer_end = time.perf_counter()
#                     return Response("ERROR",item.vector_of_sol,round(timer_end-timer_start,6),0,all_steps,"Zero diagonal element encountered")

#                 L[i][j] = (item.matrix[i][j] - sum_val) / L[j][j]
#                 addsteps(all_steps,f"L({i},{j}) = (A({i},{j}) - {sum_val})/L({j},{j})",L,item.vector_of_sol,L)

#     # --- Step 2: Solve Ly = b using Forward Substitution ---
#     # L is the lower triangular matrix

#     y = [Decimal("0") for _ in range(item.size)]
#     for i in range(item.size):
#         sum_y = Decimal("0")
#         for j in range(i):
#             sum_y += L[i][j] * y[j]

#         # Since L[i][i] is calculated and stored, we must divide by it here
#         # (L[i][i] should never be zero for positive-definite matrices)
#         if L[i][i] == 0:
#             timer_end = time.perf_counter()
#             addsteps(all_steps,"check for diagonals = zero",L,item.vector_of_sol,L)
#             return Response("ERROR",item.vector_of_sol,round(timer_end-timer_start,6),0,all_steps,f"L{i},{i} zero during forward substitution")

#         y[i] = (item.vector_of_sol[i] - sum_y) / L[i][i]
#         addsteps(all_steps,f"y({i}) = (b({i}) - {sum_y})/L({i},{i})",L,y,L)

#     # --- Step 3: Solve L^T x = y using Backward Substitution ---
#     # L^T is the upper triangular matrix

#     x = [Decimal("0") for _ in range(item.size)]
#     for i in range(item.size - 1, -1, -1):
#         sum_x = Decimal("0")
#         for j in range(i + 1, item.size):
#             # L^T[i][j] is L[j][i]
#             sum_x += L[j][i] * x[j]

#         # The pivot L^T[i][i] is L[i][i]
#         if L[i][i] == 0:
#             timer_end = time.perf_counter()
#             addsteps(all_steps, "check for diagonals = zero", L, item.vector_of_sol, L)
#             return Response("ERROR",item.vector_of_sol,round(timer_end-timer_start,6),0,all_steps,f"L{i},{i} zero during forward substitution")

#         x[i] = (y[i] - sum_x) / L[i][i]
#         addsteps(all_steps, f"x({i}) = (y({i}) - {sum_x})/L({i},{i})", L, x, L)
#     timer_end = time.perf_counter()
#     return Response("SUCCESS",x,round(timer_end-timer_start,6),0,all_steps,"",L)
def LU_decomposition_Cholesky_method(item: Item, all_steps: List['Steps']):
    """
    Cholesky decomposition A = L * L^T
    Only works for symmetric positive-definite matrices.
    """

    timer_start = time.perf_counter()
    n = item.size

    # Create L matrix full of zeros
    L = [[Decimal("0") for _ in range(n)] for _ in range(n)]

    # -------------------------------
    # STEP 1 — Cholesky Factorization
    # -------------------------------
    for i in range(n):

        # ---- Compute L[i][i]  ----
        # l_ii = √( a_ii − sum_{j=1→i-1} l_ij^2 )
        sum_sq = Decimal("0")
        for j in range(i):
            sum_sq += L[i][j] * L[i][j]

        diag_value = item.matrix[i][i] - sum_sq

        if diag_value <= 0:
            timer_end = time.perf_counter()
            addsteps(all_steps,
                     f"A({i},{i}) - Σ(l[{i}][j]^2) must be > 0. Got {diag_value}",
                     L, item.vector_of_sol)
            return Response("ERROR", item.vector_of_sol,
                            round(timer_end - timer_start, 6), 0, all_steps,
                            "Matrix is NOT positive definite")

        L[i][i] = diag_value.sqrt()

        addsteps(all_steps,
                 f"L({i},{i}) = √( A({i},{i}) - {sum_sq} )",
                 L, item.vector_of_sol, L,pivotIndex={"r":i,"c":i},highlightRow=i)

        # ---- Compute L[k][i] for k=i+1 .. n ----
        for k in range(i+1, n):
            # l_ki = ( a_ki − Σ(l_kj * l_ij) ) / l_ii
            sum_prod = Decimal("0")
            for j in range(i):
                sum_prod += L[k][j] * L[i][j]

            L[k][i] = (item.matrix[k][i] - sum_prod) / L[i][i]

            addsteps(all_steps,
                     f"L({k},{i}) = ( A({k},{i}) − {sum_prod} ) / L({i},{i})",
                     L, item.vector_of_sol, L,pivotIndex={"r":k,"c":i})

    # -------------------------------
    # STEP 2 — Forward substitution Ly = b
    # -------------------------------
    y = [Decimal("0") for _ in range(n)]
    for i in range(n):
        s = Decimal("0")
        for j in range(i):
            s += L[i][j] * y[j]

        y[i] = (item.vector_of_sol[i] - s) / L[i][i]

        addsteps(all_steps,
                 f"y({i}) = ( b({i}) − {s} ) / L({i},{i})",
                 L, y, L,pivotIndex={"r":i,"c":i},highlightRow=i)

    # -------------------------------
    # STEP 3 — Backward substitution L^T x = y
    # -------------------------------
    x = [Decimal("0") for _ in range(n)]
    for i in range(n - 1, -1, -1):
        s = Decimal("0")
        for j in range(i + 1, n):
            s += L[j][i] * x[j]

        x[i] = (y[i] - s) / L[i][i]

        addsteps(all_steps,
                 f"x({i}) = ( y({i}) − {s} ) / L({i},{i})",
                 L, x, L,pivotIndex={"r":i,"c":i},highlightRow=i)

    timer_end = time.perf_counter()
    return Response("SUCCESS", x,
                    round(timer_end - timer_start, 6), 0, all_steps, "", L)
