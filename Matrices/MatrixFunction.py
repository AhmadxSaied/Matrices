from decimal import Decimal, getcontext
class MatrixFunction:
    """
    If there is any bugs please try to solve them and tell me
    """
    def __init__(self,Precision):
        self.Precision = Precision
        getcontext().prec = self.Precision
    @staticmethod    
    def format_M(self,M):
        try:
            return [[round(x, self.Precision) for x in row] for row in M]
        except:
            return [round(x, self.Precision) for x in M]
    @staticmethod
    def naive_Gauss(Matrix):
        Matrix_copy = Matrix  # Dont forget that naive gauss doesnt deal with division be zeros so we must be prepared to catch the Division by zero error.
        try:
            # first loop is for the rows
            for i in range(len(Matrix_copy)):
                # Second loop to get the row under the position im in to perform gauss forward elemenation on the element under the pivot
                for j in range(i + 1, len(Matrix_copy)):
                    # We get the Multiplier and as we know the value of the new row jxk is Each element in the row below pivot substracted from the multiplier multiplied by each element in the row of the pivot
                    Multiplier = Matrix_copy[j][i] / Matrix_copy[i][i]
                    for k in range(len(Matrix_copy[i])):
                        Matrix_copy[j][k] = (Matrix_copy[i][k] * Multiplier * -1) + Matrix_copy[j][k]
                    # There are three loops so we can say its O(n^3)
        except Exception:
            print("Division by Zero Happend")
        return Matrix_copy

    @staticmethod
    def BackwardSubstitution(Matrix):
        Sol = [None] * len(Matrix)
        try:
            # Here we calculate the nxn element
            XLast = Matrix[len(Matrix) - 1][len(Matrix)] / Matrix[len(Matrix) - 1][len(Matrix) - 1]
            Sol[len(Matrix) - 1] = XLast

            # We want the elementof the second to last row then the element in ixj were j is the element in the column that we just previously calculated
            # We carry out the formula in the lecture B(solution of equation) - sum(Each X i calculated multiplied by there coefficient) in the equation and all of that divided by the coefficient of the Current X im calculating
            for i in range(len(Matrix) - 2, -1, -1):
                sum_val = 0.0
                for j in range(i + 1, len(Matrix)):
                    sum_val += Sol[j] * Matrix[i][j]
                Sol[i] = (Matrix[i][len(Matrix)] - sum_val) / Matrix[i][i]
                if not float(Sol[i]) == float("inf") and (Sol[i] != Sol[i]):
                    raise Exception()
            # There are Two loops so we can say that its O(n^2)

        except Exception:
            print("Division by Zero")
        return Sol

    @staticmethod
    def naive_Gauss_With_Pivot(Matrix):
        Matrix_copy = Matrix
        try:
            for i in range(len(Matrix_copy)):
                # Here We want to get the Highest Value in The column we are currently in And Strore its Row index After that we want to swap the row we are currently pointing to to the Row With the highest magnitude pivot value
                Max = float("-inf")
                Max_index = -1
                for j in range(i, len(Matrix_copy)):
                    if abs(Matrix_copy[j][i]) > Max:
                        Max = abs(Matrix_copy[j][i])
                        Max_index = j

                # Here We assign a refrence to the row we are in And then we swap the row we are in with the row with the highest magnitude pivot We then replace the row we and in with the row we swapped And Voala we have introduced Pivoting
                Temp = Matrix_copy[i]
                Matrix_copy[i] = Matrix_copy[Max_index]
                Matrix_copy[Max_index] = Temp

                # Here we repeat what we discussed Above;
                for j in range(i + 1, len(Matrix_copy)):
                    Multiplier = Matrix_copy[j][i] / Matrix_copy[i][i]
                    for k in range(len(Matrix_copy[i])):
                        Matrix_copy[j][k] = (Matrix_copy[i][k] * Multiplier * -1.0) + Matrix_copy[j][k]

                # We can see here the Complexity is O(n(n + n^2)) Which is O(n^3)

        except Exception:
            print("Division by Zero Happend")

        return Matrix_copy

    @staticmethod
    def naive_Gauss_With_Pivot_and_Scale(Matrix, Jordan):
        Matrix_copy = Matrix
        try:
            for i in range(len(Matrix_copy)):
                # Here We will introduce Scaling We want to Normalize First then keep track of the Largest Magnitude Pivot after we normalized We will Have an array each corresponding index refers to the Normalized pivot We keep track of the index With the highest magnitude normalized pivot value and We swap Like before
                Max_index = -1
                overAllMax = float("-inf")
                scale = [0] * len(Matrix)

                for j in range(i, len(Matrix_copy)):
                    max_val = float("-inf")
                    for k in range(j, len(Matrix_copy[0]) - 1):
                        if Matrix_copy[j][k] > max_val:
                            max_val = Matrix_copy[j][k]

                    scale[j] = abs(Matrix_copy[j][j] / max_val)
                    if scale[j] > overAllMax:
                        overAllMax = scale[j]
                        Max_index = j

                Temp = Matrix_copy[i]
                Matrix_copy[i] = Matrix_copy[Max_index]
                Matrix_copy[Max_index] = Temp

                # Here is the same as before nothing new
                for j in range(0 if Jordan else i + 1, len(Matrix_copy)):
                    if Jordan and i == j:
                        continue
                    Multiplier = Matrix_copy[j][i] / Matrix_copy[i][i]
                    for k in range(len(Matrix_copy[i])):
                        Matrix_copy[j][k] = (Matrix_copy[i][k] * Multiplier * -1.0) + Matrix_copy[j][k]

                # Complexity is O(n(n^2 + n^2)) Which is O(n^3)
        except Exception:
            print("Division by Zero Happend")

        return Matrix_copy

    @staticmethod
    def Crout_LU(Matrix):
        # We Firstly Initializa the L and U Matrices
        L = [[None] * len(Matrix[0]) for _ in range(len(Matrix))]
        U = [[None] * len(Matrix[0]) for _ in range(len(Matrix))]

        # The Loops are appling A Bunch of rules...
        for i in range(len(Matrix)):
            for j in range(len(Matrix[0])):
                if i == j:
                    U[i][j] = 1.0
                    if j != 0:
                        sum_val = 0.0
                        for k in range(j):
                            sum_val += L[i][k] * U[k][i]
                        L[i][j] = Matrix[i][j] - sum_val

                if i < j:
                    L[i][j] = 0.0
                if j < i:
                    U[i][j] = 0.0
                if j == 0:
                    L[i][0] = Matrix[i][0]
                if i > j and j != 0:
                    sum_val = 0.0
                    for k in range(j):
                        sum_val += L[i][k] * U[k][j]
                    L[i][j] = Matrix[i][j] - sum_val

                if j > i:
                    sum_val = 0.0
                    for k in range(i):
                        sum_val += L[i][k] * U[k][j]
                    U[i][j] = (Matrix[i][j] - sum_val) / L[i][i]

        LU = [L, U]
        return LU

    @staticmethod
    def LU(Matrix):
        # This is the normal procedure To get the LU...
        L = [[0.0] * len(Matrix) for _ in range(len(Matrix))]
        for i in range(len(Matrix)):
            L[i][i] = 1.0
            for k in range(i + 1, len(Matrix)):
                Multiplier = Matrix[k][i] / Matrix[i][i]
                for j in range(len(Matrix[i])):
                    Matrix[k][j] = (Matrix[i][j] * -1.0 * Multiplier) + Matrix[k][j]
                L[k][i] = Multiplier
                L[k][k] = 1.0

        return [L, Matrix]

    @staticmethod
    def LU_Solve(L, U, b):
        # We Solve LU System by carrying out Forward Substitution...
        Y = [None] * len(b)
        for i in range(len(L)):
            sum_val = 0.0
            for j in range(i):
                sum_val += L[i][j] * Y[j]
            Y[i] = (b[i] - sum_val) / L[i][i]

        Sol = [None] * len(b)
        Sol[len(b) - 1] = Y[-1] / U[-1][len(U) - 1]

        for i in range(len(U) - 2, -1, -1):
            sum_val = 0.0
            for j in range(i + 1, len(U[i])):
                sum_val += U[i][j] * Sol[j]
            Sol[i] = (Y[i] - sum_val) / U[i][i]

        return Sol

    @staticmethod
    def Cholesky(Matrix):
        # For Cholesky Decomposition we Have Two rules...
        L = [[None] * len(Matrix[0]) for _ in range(len(Matrix))]

        for i in range(len(Matrix)):
            for j in range(len(Matrix)):
                if j > i:
                    L[i][j] = 0.0
                else:
                    sum_val = 0.0
                    for k in range(j):
                        if i == j:
                            sum_val += (L[i][k] ** 2)
                        else:
                            sum_val += L[i][k] * L[j][k]

                    if i == j:
                        if Matrix[i][j] - sum_val < 0:
                            raise Exception("Not Positive Defenite")
                        L[i][j] = (Matrix[i][j] - sum_val) ** 0.5
                    else:
                        L[i][j] = (Matrix[i][j] - sum_val) / L[j][j]

        return L

    @staticmethod
    def Matrix_Multiplication(Matrix_1, Matrix_2):
        # Function For Multiplying Two Matrices
        if len(Matrix_1[0]) == len(Matrix_2):
            Result = [[None] * len(Matrix_2[0]) for _ in range(len(Matrix_1))]
            for i in range(len(Matrix_1)):
                for j in range(len(Matrix_2[0])):
                    Sum = 0.0
                    for k in range(len(Matrix_2)):
                        Sum += Matrix_1[i][k] * Matrix_2[k][j]
                    Result[i][j] = Sum
            return Result
        return None

    def to_decimal_matrix(M):
        return [[Decimal(str(x)) for x in row] for row in M]

    @staticmethod
    def Matrix_Subtraction(Matrix_1, Matrix_2):
        # Function for Subtracting Two Matrices
        if len(Matrix_1) == len(Matrix_2) and len(Matrix_1[0]) == len(Matrix_2[0]):
            Result = [[None] * len(Matrix_1[0]) for _ in range(len(Matrix_1))]
            for i in range(len(Matrix_1)):
                for j in range(len(Matrix_1[0])):
                    Result[i][j] = Matrix_1[i][j] - Matrix_2[i][j]
            return Result
        return None

    @staticmethod
    def Transpose(Matrix_1):
        # Calculating the Transpose of the Matrix
        Result = [[None] * len(Matrix_1) for _ in range(len(Matrix_1[0]))]
        for i in range(len(Matrix_1)):
            for j in range(len(Matrix_1[0])):
                Result[j][i] = Matrix_1[i][j]
        return Result

    @staticmethod
    def Scaler(Matrix_1, Scaler):
        # Multiplying A matrix By a Scaler
        Result = [[None] * len(Matrix_1[0]) for _ in range(len(Matrix_1))]
        for i in range(len(Matrix_1)):
            for j in range(len(Matrix_1[0])):
                Result[i][j] = Matrix_1[i][j] * Scaler
        return Result

    class Eigen:
        def __init__(self, EigenValue, EigenVector):
            self.EigenValue = EigenValue
            self.EigenVector = EigenVector

        def __str__(self):
            Expression = "{EigenValue: " + str(self.EigenValue)
            Expression += "\nEigenVector: " + self.EigenVector + "\n\n"
            return Expression

    def Jacobi_GaussIteration(self,Matrix, EquationResult, Tolerance, Gauss):
        Matrixx = self.to_decimal_matrix(M=Matrix)
        Sol = [None] * len(Matrixx[0])
        if len(EquationResult) == len(Matrixx):
            Max_Error = float("inf")
            IterationSol = [1.0] * len(Matrixx[0])
            for i in range(len(Sol)):
                Sol[i] = Decimal(1.0)

            while Max_Error > Tolerance:
                Error = 0.0
                for i in range(len(Matrixx)):
                    Sum = Decimal(EquationResult[i])
                    for j in range(len(Matrixx[0])):
                        if j != i:
                            Sum -= Decimal(Sol[j]) * Matrixx[i][j]
                    IterationSol[i] = Sum / Matrixx[i][i]
                    Error = max(Error, abs((IterationSol[i] - Sol[i]) / IterationSol[i]))
                    if float("inf") == Error:
                        Error = 1.0
                    if Gauss == True:
                        Sol[i] = IterationSol[i]

                Max_Error = min(Max_Error, Error)
                if Gauss == False:
                    for i in range(len(Sol)):
                        Sol[i] = IterationSol[i]
        else:
            raise Exception()
        return Sol
    @staticmethod
    def to_decimal_matrix(M):
        return [[Decimal(x) for x in row] for row in M]
if __name__== "__main__":
    Matrix = [
            [5.0,-1.0,1.0],
            [2.0,8.0,-1.0],
            [-1.0,1.0,3.0],
    ]
    EquationResult = [10.0,10.0,10.0]
    MatrixFunctions = MatrixFunction(20)
    print(MatrixFunctions.Jacobi_GaussIteration(Matrix, EquationResult, 1e-9,True))


