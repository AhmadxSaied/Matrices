from decimal import Decimal, getcontext
from pydantic import BaseModel
import time
import response
import helper


# Defines the structure of the JSON data we expect to receive.
class Item(BaseModel):
    precision: int | None = 10
    size: int
    matrix: list[list[Decimal]]
    vector_of_sol: list[Decimal]
    initial_guess: list[Decimal] = []





# +++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++ methods ++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++
# ! Naive gauss elimination
def Naive_gauss_elimination(item: Item):
    getcontext().prec = item.precision if item.precision is not None else 10
    if helper.forward_elimination_withoutPivoting(item.size, item.matrix, item.vector_of_sol) == "error":
        return "error"
    vector_of_unknowns = helper.backward_substitution(item.size, item.matrix, item.vector_of_sol)
    return vector_of_unknowns

# ! Gauss Elimination with Partial Pivoting
def Gauss_elimination_with_partial_pivoting(item: Item):
    getcontext().prec = item.precision if item.precision is not None else 10
    if helper.forward_elimination_withPivoting(item.size, item.matrix, item.vector_of_sol) == "error":
        return "error"
    vector_of_unknowns = helper.backward_substitution(item.size, item.matrix, item.vector_of_sol)
    return vector_of_unknowns

# ! Gauss Elimination with Partial Pivoting and scaling
def Gauss_elimination_with_partial_pivoting_and_scaling(item: Item):
    getcontext().prec = item.precision if item.precision is not None else 10
    if helper.forward_elimination_withPivoting_and_scaling(item.size, item.matrix, item.vector_of_sol) == "error":
        return "error"
    vector_of_unknowns = helper.backward_substitution(item.size, item.matrix, item.vector_of_sol)
    return vector_of_unknowns

# ! Gauss-Jordan elimination (with partial pivoting)
def Gauss_Jordan_elimination(item: Item):
    getcontext().prec = item.precision if item.precision is not None else 10
    # Loop over columns/pivots
    for pivot in range(item.size):
        # 1. Partial Pivoting (Find largest ABSOLUTE magnitude)
        max_val = Decimal("-1")
        max_row_index = pivot
        # Search for the largest element from the current pivot row downwards
        for row in range(pivot, item.size):
            current_abs_val = abs(item.matrix[row][pivot])
            if current_abs_val > max_val:
                max_val = current_abs_val
                max_row_index = row
        # Check for singularity
        if max_val == 0:
            return "error"  # Singular matrix
        # --- Perform Row Swap ---
        if max_row_index != pivot:
            # Swap rows in the matrix A
            item.matrix[pivot], item.matrix[max_row_index] = item.matrix[max_row_index], item.matrix[pivot]
            # Swap corresponding elements in the solution vector b
            item.vector_of_sol[pivot], item.vector_of_sol[max_row_index] = item.vector_of_sol[max_row_index], item.vector_of_sol[pivot]

        # 2. Normalization (Make the pivot element equal to 1)
        pivot_value = item.matrix[pivot][pivot]
        # Normalize the pivot row elements (from current column to the end)
        for eir in range(pivot, item.size):
            item.matrix[pivot][eir] /= pivot_value
        # Normalize the corresponding element in the solution vector
        item.vector_of_sol[pivot] /= pivot_value

        # 3. Elimination (Zero out all other entries in the pivot column)
        # rup == row index to update
        for rup in range(item.size):
            # Skip the pivot row itself
            if rup == pivot:
                continue

            # m == multiplier
            m = item.matrix[rup][pivot]
            # Eliminate elements in the matrix A (from current column to the end)
            for eir in range(pivot, item.size):
                item.matrix[rup][eir] -= m * item.matrix[pivot][eir]
            # Eliminate element in the solution vector b
            item.vector_of_sol[rup] -= m * item.vector_of_sol[pivot]
    return item.vector_of_sol

# ! Gauss-Seidel Method (without pivoting)
def Gauss_Seidel_method(item: Item):
    getcontext().prec = item.precision if item.precision is not None else 10
    values_of_unknowns = item.initial_guess[:]
    # Check for zero pivots (diagonal elements) before starting
    for i in range(item.size):
        if item.matrix[i][i] == 0:
            return "error"

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
        if max_relative_error < item.tolerance:
            # Convergence achieved
            return values_of_unknowns
    
    return {"error":f"error: Did not converge within {item.max_iterations} iterations. Final error: {max_relative_error}" ,"diagonally_dominant":item.check_diagonally_dominant(item.size, item.matrix)}

# ! Jacobi Method (without pivoting)
def Jacobi_method(item: Item, max_iterations=50, tolerance=Decimal("1e-6")):
    getcontext().prec = item.precision if item.precision is not None else 10
    values_of_unknowns = item.initial_guess[:]
    # Check for zero pivots (diagonal elements) before starting
    for i in range(item.size):
        if item.matrix[i][i] == 0:
            return "error"

    for k in range(max_iterations):
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
        if max_relative_error < tolerance:
            # Convergence achieved
            return values_of_unknowns
    return {"error":f"error: Did not converge within {max_iterations} iterations. Final error: {max_relative_error}" ,"diagonally_dominant":check_diagonally_dominant(item.size, item.matrix)}

# ! LU Decomposition Method (Doolittle's Method)
def LU_decomposition_Doolittle_method(item: Item):
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
def LU_decomposition_Crout_method(item: Item):
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
def LU_decomposition_Cholesky_method(item: Item):
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