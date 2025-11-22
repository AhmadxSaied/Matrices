from decimal import Decimal, getcontext
from pydantic import BaseModel
from response import addsteps,Steps
from typing import List
# ! forward elimination (with pivoting and scalling)
# modify in original matrix
def forward_elimination_withPivoting_and_scaling(size, matrix, vector_of_sol, steps: List['Steps']) -> str|None:
    # Loop over pivots (except last row)
    for pivot in range(size-1):
        # --- 1. Partial Pivoting (Find largest ABSOLUTE magnitude) ---
        # Initialize with a value smaller than any possible absolute magnitude
        max_val = Decimal("-1")
        max_row_index = pivot   # Initialize the index of the best pivot row
        # Search from the current pivot row down to the last row
        for row in range(pivot, size):
            row_max = max(matrix[row], key=abs)
            # Skip row if it is all zeros
            if row_max == 0:
                continue
            # The standard for Partial Pivoting is to look for the largest absolute value
            current_abs_val = abs(matrix[row][pivot] / row_max)  # ? scaling
            if current_abs_val > max_val:
                max_val = current_abs_val
                max_row_index = row
        # Check if the largest element found is zero
        if max_val == 0:
            # The matrix is singular or near-singular, and the system cannot be solved
            addsteps(steps,"The matrix is singular or near-singular, and the system cannot be solved",matrix,vector_of_sol)
            return "error"
            #return "error"

        # --- 2. Perform Row Swap ---
        # Swap rows only if the best pivot row is different from the current pivot row
        if max_row_index != pivot:
            # Swap rows in the matrix A
            dummy_A = matrix[pivot]
            matrix[pivot] = matrix[max_row_index]
            matrix[max_row_index] = dummy_A
            # CRITICAL: Swap corresponding elements in the solution vector b
            dummy_b = vector_of_sol[pivot]
            vector_of_sol[pivot] = vector_of_sol[max_row_index]
            vector_of_sol[max_row_index] = dummy_b
            addsteps(steps,f"sawp R{pivot+1} with R{max_row_index+1}",matrix,vector_of_sol)
        # rup == rows under pivot <-- row
        for rup in range(pivot+1, size):
            # m == multiplier
            m = matrix[rup][pivot] / matrix[pivot][pivot]
            # eir == elements in row  <-- col
            for eir in range(pivot, size):
                matrix[rup][eir] = matrix[rup][eir] - m * matrix[pivot][eir]
            vector_of_sol[rup] = vector_of_sol[rup] - m * vector_of_sol[pivot]
            addsteps(steps,f"R{rup+1} = R{rup+1}-({m}) * R{pivot+1} (Elimination)",matrix,vector_of_sol)
    return None
        

# ! forward elimination (with pivoting)
# modify in original matrix
def forward_elimination_withPivoting(size, matrix, vector_of_sol,steps:List['Steps']) -> str|None:
    # Loop over pivots (except last row)

    for pivot in range(size-1):

        # --- 1. Partial Pivoting (Find largest ABSOLUTE magnitude) ---
        # Initialize with a value smaller than any possible absolute magnitude
        max_val = Decimal("-1")
        max_row_index = pivot   # Initialize the index of the best pivot row
        # Search from the current pivot row down to the last row
        for row in range(pivot, size):
            # The standard for Partial Pivoting is to look for the largest absolute value
            current_abs_val = abs(matrix[row][pivot])
            if current_abs_val > max_val:
                max_val = current_abs_val
                max_row_index = row
        # Check if the largest element found is zero
        if max_val == 0:
            # The matrix is singular or near-singular, and the system cannot be solved
            addsteps(steps,"the matrix is singular or near-singular the system cannot be solved",matrix,vector_of_sol)
            return "error"

        # --- 2. Perform Row Swap ---
        # Swap rows only if the best pivot row is different from the current pivot row
        if max_row_index != pivot:
            # Swap rows in the matrix A
            dummy_A = matrix[pivot]
            matrix[pivot] = matrix[max_row_index]
            matrix[max_row_index] = dummy_A
            # CRITICAL: Swap corresponding elements in the solution vector b
            dummy_b = vector_of_sol[pivot]
            vector_of_sol[pivot] = vector_of_sol[max_row_index]
            vector_of_sol[max_row_index] = dummy_b
            addsteps(steps,f"sawp R{pivot+1} with R{max_row_index+1}",matrix,vector_of_sol)
        else:
            addsteps(steps,"there is no need for pivoting as the largest pivot is in the correct position",matrix,vector_of_sol)
        # rup == rows under pivot <-- row
        for rup in range(pivot+1, size):
            # m == multiplier
            m = matrix[rup][pivot] / matrix[pivot][pivot]
            # eir == elements in row  <-- col
            for eir in range(pivot, size):
                matrix[rup][eir] = matrix[rup][eir] - m * matrix[pivot][eir]
            vector_of_sol[rup] = vector_of_sol[rup] - m * vector_of_sol[pivot]
            if(m != 0):
                addsteps(steps,f"R{rup+1} = R{rup+1}-({m}) * R{pivot+1} (Elimination)",matrix,vector_of_sol)
    return None

# ! forward elimination (without pivoting) & store multipliers
# modify in original matrix
def forward_elimination_withoutPivoting(size, matrix, vector_of_sol,steps:List['Steps'])-> str|None:
    array_of_multipliers = []
    # Loop over pivots (except last row)
    for pivot in range(size-1):
        if matrix[pivot][pivot] == 0:
            addsteps(steps,f"Can't solve it without pivoting as the pivot in R{pivot+1} = zero",matrix,vector_of_sol)
            return "error"
        
        # rup == rows under pivot <-- row
        for rup in range(pivot+1, size):
            # m == multiplier
            m = matrix[rup][pivot] / matrix[pivot][pivot]
            array_of_multipliers.append(m)
            # eir == elements in row  <-- col
            for eir in range(pivot, size):
                matrix[rup][eir] = matrix[rup][eir] - m * matrix[pivot][eir]
            vector_of_sol[rup] = vector_of_sol[rup] - m * vector_of_sol[pivot]
            addsteps(steps,f"R{rup+1} = R{rup+1}-({m}) * R{pivot+1} (Elimination)",matrix,vector_of_sol)
    return None

# ! backward substitution
def backward_substitution(size, matrix, vector_of_sol,steps:List['Steps']):
    vector_of_unknowns = [None for _ in range(size)] 
    vector_of_unknowns[size - 1] = vector_of_sol[size - 1] / matrix[size - 1][size - 1]  # find value of last unknown
    addsteps(steps,f"X{size} = {vector_of_sol[size - 1]} / {matrix[size - 1][size - 1]} = {vector_of_unknowns[size - 1]}",matrix,vector_of_unknowns)
    for pivot in range(size - 2, -1, -1):
        # s == sum
        s = 0
        # j == elements in right side of pivot
        for j in range(pivot + 1, size):
            s += vector_of_unknowns[j] * matrix[pivot][j]
        vector_of_unknowns[pivot] = (
            vector_of_sol[pivot] - s) / matrix[pivot][pivot]
        numerator = vector_of_sol[pivot] - s
        addsteps(steps,f"X{pivot+1} = ({numerator}) / {matrix[pivot][pivot]} = {vector_of_unknowns[pivot]}",matrix,vector_of_unknowns)
    return vector_of_unknowns

def check_diagonally_dominant(size,vector_of_sol ,matrix,steps:List['Steps']):
    strictly_dominant = False
    for row in range(size):
        diag = abs(matrix[row][row])
        row_sum = sum(abs(matrix[row][col])
                      for col in range(size) if col != row)
        if diag < row_sum:        # violates diagonal dominance
            return False
        if diag > row_sum:        # at least one strict dominance
            strictly_dominant = True
    addsteps(steps,f"diagonally dominant -> {strictly_dominant}",matrix,vector_of_sol)
    return strictly_dominant
def check_havesol(size,vector_of_sol,matrix,steps:List['Steps']) -> bool:
    has_zero_pivot=False
    for i in range(size):
        print(matrix[i][i],vector_of_sol[i])
        if(matrix[i][i] == Decimal("0")):
            has_zero_pivot=True
            zero_pivot_row = i
            if(vector_of_sol[i] != Decimal("0")):
                addsteps(steps,f"Fatal Error: R{i+1} results in 0 = {vector_of_sol[i]} System has no solution",matrix,vector_of_sol)
                return "None"
    if has_zero_pivot:
        addsteps(steps,f"the System is consistent but the System has infinite number of solution",matrix,vector_of_sol)
        return "Infinite"
    addsteps(steps,"has a unique solution ",matrix,vector_of_sol)
    return "unique"
def backward_elimination(size,vector_of_sol,matrix,steps:List['Steps']):
    vector_of_sol[size-1] = vector_of_sol[size-1]/matrix[size-1][size-1]
    matrix[size-1][size-1] = 1
    for pivot in range(size-1,-1,-1):
        for rup in range(pivot):
            m = matrix[rup][pivot]

            for eir in range(pivot,size):
                matrix[rup][eir] -= m*matrix[pivot][eir]
            vector_of_sol[rup]-=m* vector_of_sol[pivot]
            if(m != Decimal("0")):
                addsteps(steps,f"R{rup+1} = R{rup+1} - ({m}) * R{pivot+1}",matrix,vector_of_sol)
def normalize_matrix(size,vector_of_sol,matrix,steps:List['Steps']):
    for i in range(size):
        pivot_value = matrix[i][i]
        
        if pivot_value == 0:
            # If a zero pivot remains here, it means the system is singular.
            addsteps(steps, f"Error: Zero pivot detected at R{i+1} during normalization.", matrix, vector_of_sol)
            return "error"

        # Divide the entire row by the pivot value
        for j in range(i, size):
            matrix[i][j] /= pivot_value
        vector_of_sol[i] /= pivot_value
        
        # Tracking: Log the normalization
        addsteps(steps,f"Normalized R{i+1} by dividing by pivot ({pivot_value})",matrix, vector_of_sol)
        
    return None # Success