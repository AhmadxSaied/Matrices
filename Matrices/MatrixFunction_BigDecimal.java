package Matrices;

import java.util.ArrayList;
import java.util.List;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

public class MatrixFunction_BigDecimal {
    private static int Scale = 10;
    private static RoundingMode Mode = RoundingMode.HALF_UP;

    /*
     * If there is any bugs please try to solve them and tell me
     */
    public static BigDecimal[][] naive_Gauss(Double Matrix[][]) {
        BigDecimal Matrix_copy[][] = new BigDecimal[Matrix.length][Matrix[0].length];
        /*
         * Dont forget that naive gauss doesnt deal with division be zeros so we
         * must be prepared to catch the Division by zero error.
         */
        for (int i = 0; i < Matrix_copy.length; i++) {
            for (int j = 0; j < Matrix_copy[i].length; j++) {
                Matrix_copy[i][j] = BigDecimal.valueOf(Matrix[i][j]).setScale(Scale);
            }
        }
        try {
            /*
             * first loop is for the rows
             */
            for (int i = 0; i < Matrix_copy.length; i++) {
                /*
                 * Second loop to get the row under the position im in to
                 * perform gauss forward elemenation on the element under the pivot
                 */
                for (int j = i + 1; j < Matrix_copy.length; j++) {
                    /*
                     * We get the Multiplier
                     * and as we know the value of the new row jxk is
                     * Each element in the row below pivot substracted from
                     * the multiplier multiplied by each element in the row of the pivot
                     */
                    BigDecimal Multiplier = Matrix_copy[j][i].divide(Matrix_copy[i][i], Scale, Mode);

                    for (int k = 0; k < Matrix_copy[i].length; k++) {
                        Matrix_copy[j][k] = Matrix_copy[i][k].multiply(Multiplier.multiply(BigDecimal.valueOf(-1.0)))
                                .add(Matrix_copy[j][k]).setScale(Scale, Mode);
                    }
                    /*
                     * There are three loops so we can say its O(n^3)
                     */
                }
            }
        } catch (Error e) {
            System.out.println("Division by Zero Happend");
        }
        return Matrix_copy;
    }

    public static Double[] BackwardSubstitution(Double Matrix[][]) {
        Double Sol[] = new Double[Matrix.length];
        try {
            // Here we calculate the nxn element
            double XLast = Matrix[Matrix.length - 1][Matrix.length] / Matrix[Matrix.length - 1][Matrix.length - 1];
            Sol[Matrix.length - 1] = XLast;
            /*
             * We want the elementof the second to last row
             * then the element in ixj were j is the element
             * in the column that we just previously calculated
             * 
             */
            /*
             * We carry out the formula in the lecture
             * B(solution of equation) - sum(Each X i calculated multiplied by there
             * coefficient)
             * in the equation and all of that divided by the coefficient of the Current X
             * im calculating
             */
            for (int i = Matrix.length - 2; i >= 0; i--) {
                double sum = 0.0;
                for (int j = i + 1; j <= Matrix.length - 1; j++) {
                    sum += Sol[j] * Matrix[i][j];
                }
                Sol[i] = (Matrix[i][Matrix.length] - sum) / Matrix[i][i];
                if (!Sol[i].isInfinite() && Sol[i].isNaN()) {
                    throw new Error();
                }
            }
            /*
             * There are Two loops so we can say that its O(n^2)
             */

        } catch (Error e) {
            System.out.println("Division by Zero");
        }
        return Sol;
    }

    public static BigDecimal[][] naive_Gauss_With_Pivot(Double Matrix[][]) {
        BigDecimal Matrix_copy[][] = new BigDecimal[Matrix.length][Matrix[0].length];
        for (int i = 0; i < Matrix_copy.length; i++) {
            for (int j = 0; j < Matrix_copy[i].length; j++) {
                Matrix_copy[i][j] = BigDecimal.valueOf(Matrix[i][j]).setScale(Scale, Mode);
            }
        }
        try {
            for (int i = 0; i < Matrix_copy.length; i++) {
                /*
                 * Here We want to get the Highest Value in The column we are currently in
                 * And Strore its Row index
                 * After that we want to swap the row we are currently pointing to
                 * to the Row With the highest magnitude pivot value
                 */
                BigDecimal Max = BigDecimal.valueOf(-1000000000000.0).setScale(Scale);
                int Max_index = -1;
                for (int j = i; j < Matrix_copy.length; j++) {
                    if ((Matrix_copy[j][i]).abs().compareTo(Max) == 1) {
                        Max = Matrix_copy[j][i].abs();
                        Max_index = j;
                    }
                }
                /*
                 * Here We assign a refrence to the row we are in
                 * And then we swap the row we are in with the row with the highest magnitude
                 * pivot We then replace the row we and in with the row we swapped
                 * And Voala we have introduced Pivoting
                 */
                BigDecimal Temp[] = Matrix_copy[i];
                Matrix_copy[i] = Matrix_copy[Max_index];
                Matrix_copy[Max_index] = Temp;
                /*
                 * Here we repeat what we discussed Above;
                 */
                for (int j = i + 1; j < Matrix_copy.length; j++) {
                    BigDecimal Multiplier = (Matrix_copy[j][i]).divide((Matrix_copy[i][i]), Scale, Mode);
                    for (int k = 0; k < Matrix_copy[i].length; k++) {
                        Matrix_copy[j][k] = (Matrix_copy[i][k].multiply(Multiplier.multiply(BigDecimal.valueOf(-1.0))))
                                .add(Matrix_copy[j][k]).setScale(Scale, Mode);
                    }
                }
                /*
                 * We can see here the Complexity is O(n(n + n^2)) Which is O(n^3)
                 */
            }
        } catch (Error e) {
            System.out.println("Division by Zero Happend");
        }
        return Matrix_copy;
    }

    public static BigDecimal[][] naive_Gauss_With_Pivot_and_Scale(Double Matrix[][], Boolean Jordan) {
        BigDecimal Matrix_copy[][] = new BigDecimal[Matrix.length][Matrix[0].length];
        for (int i = 0; i < Matrix_copy.length; i++) {
            for (int j = 0; j < Matrix_copy[i].length; j++) {
                Matrix_copy[i][j] = BigDecimal.valueOf(Matrix[i][j]).setScale(Scale, Mode);
            }
        }
        try {
            for (int i = 0; i < Matrix_copy.length; i++) {
                /*
                 * Here We will introduce Scaling
                 * We want to Normalize First then keep track of the Largest Magnitude
                 * Pivot after we normalized
                 * We will Have an array each corresponding index refers to the
                 * Normalized pivot
                 * We keep track of the index With the highest magnitude normalized
                 * pivot value
                 * and We swap Like before
                 */
                int Max_index = -1;
                BigDecimal overAllMax = BigDecimal.valueOf(-1000000000.0).setScale(Scale, Mode);
                BigDecimal scale[] = new BigDecimal[Matrix.length];
                for (int j = i; j < Matrix_copy.length; j++) {
                    BigDecimal max = BigDecimal.valueOf(-1000000000.0).setScale(Scale, Mode);
                    for (int k = j; k < Matrix_copy[0].length - 1; k++) {
                        if (Matrix_copy[j][k].compareTo(max) == 1) {
                            max = Matrix_copy[j][k].setScale(Scale, Mode);
                        }
                    }
                    /*
                     * Here We calculate the scale which is pivot/ max value in row
                     */
                    scale[j] = Matrix_copy[j][j].divide(max, Scale, Mode)
                            .abs();
                    if (scale[j].compareTo(overAllMax) == 1) {
                        overAllMax = scale[j].setScale(Scale, Mode);
                        Max_index = j;
                    }
                }
                BigDecimal Temp[] = Matrix_copy[i];
                Matrix_copy[i] = Matrix_copy[Max_index];
                Matrix_copy[Max_index] = Temp;
                /*
                 * Here is the same as before nothing new
                 */
                for (int j = Jordan ? 0 : i + 1; j < Matrix_copy.length; j++) {
                    if (Jordan && i == j) {
                        continue;
                    }
                    BigDecimal Multiplier = (Matrix_copy[j][i]).divide((Matrix_copy[i][i]), Scale, Mode);
                    for (int k = 0; k < Matrix_copy[i].length; k++) {
                        Matrix_copy[j][k] = (Matrix_copy[i][k].multiply(Multiplier.multiply(BigDecimal.valueOf(-1.0)))
                                .add(Matrix_copy[j][k])).setScale(Scale, Mode);
                    }
                }
                /*
                 * Complexity is O(n(n^2 + n^2)) Which is O(n^3)
                 */
            }
        } catch (Error e) {
            System.out.println("Division by Zero Happend");
        }
        return Matrix_copy;
    }

    public static BigDecimal[][][] Crout_LU(Double[][] Matrix) {
        BigDecimal Matrix_copy[][] = new BigDecimal[Matrix.length][Matrix[0].length];
        for (int i = 0; i < Matrix_copy.length; i++) {
            for (int j = 0; j < Matrix_copy[i].length; j++) {
                Matrix_copy[i][j] = BigDecimal.valueOf(Matrix[i][j]).setScale(Scale, Mode);
            }
        }
        /*
         * We Firstly Initializa the L and U Matrices
         */
        BigDecimal L[][] = new BigDecimal[Matrix.length][Matrix[0].length];
        BigDecimal U[][] = new BigDecimal[Matrix.length][Matrix[0].length];
        /*
         * The Loops are appling A Bunch of rules
         * For L matrix
         * 1. If you are Above the Diagonal the elements equal 0
         * 2. If You are in the First Column in the original matrix the L Value is the
         * Coressponding element in its place in the Original Matrix
         * 3. If the row number is Higher than the Columns You need to Calculate the
         * summation of the Previous element in the row with the Coressponding column in
         * U
         */
        /*
         * For U we have simple rules
         * 1. Elements below diagonal is 0
         * 2. Elements of U is The Diffrence between the element of corressponding U in
         * Original Matrix
         * and The summation of Previous U's multiplied by Corressponding L
         */
        for (int i = 0; i < Matrix_copy.length; i++) {
            for (int j = 0; j < Matrix_copy[0].length; j++) {
                if (i == j) {
                    U[i][j] = BigDecimal.valueOf(1.0).setScale(Scale, Mode);
                    if (j != 0) {
                        BigDecimal sum = BigDecimal.valueOf(0.0).setScale(Scale, Mode);
                        for (int k = 0; k < j; k++) {
                            sum = sum.add(L[i][k].multiply(U[k][i]).setScale(Scale, Mode));
                        }
                        L[i][j] = Matrix_copy[i][j].subtract(sum).setScale(Scale, Mode);
                    }

                }
                if (i < j) {
                    L[i][j] = BigDecimal.valueOf(0.0).setScale(Scale, Mode);
                }
                if (j < i) {
                    U[i][j] = BigDecimal.valueOf(0.0).setScale(Scale, Mode);
                }
                if (j == 0) {
                    L[i][0] = Matrix_copy[i][0].setScale(Scale, Mode);
                }
                if (i > j && j != 0) {
                    BigDecimal sum = BigDecimal.valueOf(0.0).setScale(Scale, Mode);
                    for (int k = 0; k < j; k++) {
                        sum = sum.add(L[i][k].multiply(U[k][j])).setScale(Scale, Mode);
                    }
                    L[i][j] = Matrix_copy[i][j].subtract(sum).setScale(Scale, Mode);
                }

                if (j > i) {
                    BigDecimal sum = BigDecimal.valueOf(0.0).setScale(Scale, Mode);
                    for (int k = 0; k < i; k++) {
                        sum = L[i][k].multiply(U[k][j]).setScale(Scale, Mode);
                    }
                    U[i][j] = (Matrix_copy[i][j].subtract(sum)).divide(L[i][i], Scale, Mode);
                }
            }
        }
        BigDecimal LU[][][] = new BigDecimal[2][Matrix.length][Matrix[0].length];
        LU[0] = L;
        LU[1] = U;
        return LU;
    }

    public static BigDecimal[][][] LU(Double[][] Matrix) {
        BigDecimal Matrix_copy[][] = new BigDecimal[Matrix.length][Matrix[0].length];
        for (int i = 0; i < Matrix_copy.length; i++) {
            for (int j = 0; j < Matrix_copy[i].length; j++) {
                Matrix_copy[i][j] = BigDecimal.valueOf(Matrix[i][j]).setScale(Scale, Mode);
            }
        }
        /*
         * This is the normal procedure To get the LU
         * We apply Forward Gauss Elimination
         * We store the Multiplier in L
         * The Resulting Matrix of elimination is U
         */
        BigDecimal L[][] = new BigDecimal[Matrix.length][Matrix.length];
        for (int i = 0; i < Matrix_copy.length; i++) {
            for (int j = 0; j < Matrix_copy.length; j++) {
                L[i][j] = BigDecimal.valueOf(0.0).setScale(Scale, Mode);
            }
        }
        for (int i = 0; i < Matrix_copy[0].length; i++) {
            L[i][i] = BigDecimal.valueOf(1.0).setScale(Scale, Mode);
            for (int k = i + 1; k < Matrix_copy.length; k++) {
                BigDecimal Multiplier = Matrix_copy[k][i].divide(Matrix_copy[i][i], Scale, Mode);
                for (int j = 0; j < Matrix[i].length; j++) {
                    Matrix_copy[k][j] = (Matrix_copy[i][j].multiply((BigDecimal.valueOf(-1.0).multiply(Multiplier)))
                            .add(Matrix_copy[k][j])).setScale(Scale, Mode);
                }
                L[k][i] = Multiplier;
                L[k][k] = BigDecimal.valueOf(1.0).setScale(Scale, Mode);
                ;
            }

        }
        BigDecimal[][][] LU = new BigDecimal[2][Matrix.length][Matrix[0].length];
        LU[0] = L;
        LU[1] = Matrix_copy;
        return LU;
    }

    public static BigDecimal[] LU_Solve(BigDecimal[][] L, BigDecimal[][] U, Double[] b) {
        BigDecimal[] b_copy = new BigDecimal[b.length];
        for (int i = 0; i < b.length; i++) {
            b_copy[i] = BigDecimal.valueOf(b[i]).setScale(Scale, Mode);
        }
        /*
         * We Solve LU System by carrying out Forward Substitution between L and b
         * We then We carry out Backward Sustitution of between U and result of previous
         * L substitution
         */
        BigDecimal[] Y = new BigDecimal[b.length];
        for (int i = 0; i < L.length; i++) {
            BigDecimal sum = BigDecimal.valueOf(0.0).setScale(Scale, Mode);
            for (int j = 0; j < i; j++) {
                sum = sum.add(L[i][j].multiply(Y[j])).setScale(Scale, Mode);
            }
            Y[i] = (b_copy[i].subtract(sum)).divide(L[i][i], Scale, Mode);
        }
        BigDecimal[] Sol = new BigDecimal[b.length];
        Sol[b.length - 1] = (Y[Y.length - 1].divide(U[Y.length - 1][Y.length - 1], Scale, Mode));
        for (int i = U.length - 2; i >= 0; i--) {

            BigDecimal sum = BigDecimal.valueOf(0.0).setScale(Scale, Mode);

            for (int j = i + 1; j < U[i].length; j++) {
                sum = sum.add(U[i][j].multiply(Sol[j])).setScale(Scale, Mode);
            }
            Sol[i] = (Y[i].subtract(sum)).divide(U[i][i], Scale, Mode);
        }
        return Sol;
    }

    public static BigDecimal[][] Cholesky(Double[][] Matrix) {
        BigDecimal Matrix_copy[][] = new BigDecimal[Matrix.length][Matrix[0].length];
        for (int i = 0; i < Matrix_copy.length; i++) {
            for (int j = 0; j < Matrix_copy[i].length; j++) {
                Matrix_copy[i][j] = BigDecimal.valueOf(Matrix[i][j]).setScale(Scale, Mode);
            }
        }
        /*
         * For Cholesky Decomposition we Have Two rules
         * If the i is less than the J meaning the element is below diagonal
         * We First need the summation of the Previous L elements and the L element of
         * the row we are Currently calculation
         * We get difference Between element in the corressponding original Matrix and
         * then Divide by the L element of diagonal of column we are in [j,j]
         * If I==J we need to get the summation of the previous elements of the row
         * Squared
         * we calculate the difference between the Corresponding element in the original
         * matrix and the Summation
         * Check That it is not a Negative Sqrt and then You have got Your L
         */
        BigDecimal[][] L = new BigDecimal[Matrix_copy.length][Matrix_copy[0].length];
        for (int i = 0; i < Matrix_copy.length; i++) {
            for (int j = 0; j < Matrix_copy.length; j++) {
                if (j > i) {
                    L[i][j] = BigDecimal.valueOf(0.0).setScale(Scale, Mode);
                } else {
                    BigDecimal sum = BigDecimal.valueOf(0.0).setScale(Scale, Mode);
                    for (int k = 0; k < j; k++) {
                        if (i == j) {
                            sum = sum.add(L[i][k].pow(2));
                        } else {
                            sum = sum.add(L[i][k].multiply(L[j][k])).setScale(Scale, Mode);
                        }
                    }
                    if (i == j) {
                        if (Matrix_copy[i][j].compareTo(sum) == -1) {
                            throw new Error("Not Positive Defenite");
                        }
                        L[i][j] = (Matrix_copy[i][j].subtract(sum))
                                .sqrt(MathContext.DECIMAL128).setScale(Scale, Mode);
                    } else {
                        L[i][j] = (Matrix_copy[i][j].subtract(sum)).divide(L[j][j], Scale, Mode);
                    }
                }
            }
        }
        return L;
    }

    public static BigDecimal[][] Matrix_Multiplication(BigDecimal[][] Matrix_1, BigDecimal[][] Matrix_2) {
        /*
         * Function For Multiplying Two Matrices
         */
        if (Matrix_1[0].length == Matrix_2.length) {
            BigDecimal Result[][] = new BigDecimal[Matrix_1.length][Matrix_2[0].length];
            for (int i = 0; i < Matrix_1.length; i++) {
                for (int j = 0; j < Matrix_2[0].length; j++) {
                    BigDecimal Sum = BigDecimal.valueOf(0.0).setScale(Scale, Mode);
                    for (int k = 0; k < Matrix_2.length; k++) {
                        Sum = Sum.add(Matrix_1[i][k].multiply(Matrix_2[k][j])).setScale(Scale, Mode);
                    }
                    Result[i][j] = Sum;
                }
            }
            return Result;
        }
        return null;
    }

    public static BigDecimal[][] Matrix_Subtraction(BigDecimal[][] Matrix_1, BigDecimal[][] Matrix_2) {
        /*
         * Function for Subtracting Two Matrices
         */
        if (Matrix_1.length == Matrix_2.length && Matrix_1[0].length == Matrix_2[0].length) {
            BigDecimal Result[][] = new BigDecimal[Matrix_2.length][Matrix_1[0].length];
            for (int i = 0; i < Matrix_1.length; i++) {
                for (int j = 0; j < Matrix_1[0].length; j++) {
                    Result[i][j] = Matrix_1[i][j].subtract(Matrix_2[i][j]).setScale(Scale, Mode);
                }
            }
            return Result;
        }
        return null;
    }

    public static BigDecimal[][] Transpose(BigDecimal[][] Matrix_1) {
        /*
         * Calculating the Transpose of the Matrix
         */
        BigDecimal Result[][] = new BigDecimal[Matrix_1[0].length][Matrix_1.length];
        for (int i = 0; i < Matrix_1.length; i++) {
            for (int j = 0; j < Matrix_1[0].length; j++) {
                Result[j][i] = Matrix_1[i][j];
            }
        }
        return Result;

    }

    public static BigDecimal[][] Scaler(BigDecimal[][] Matrix_1, BigDecimal Scaler) {
        /*
         * Multiplying A matrix By a Scaler
         */
        BigDecimal Result[][] = new BigDecimal[Matrix_1.length][Matrix_1[0].length];
        for (int i = 0; i < Matrix_1.length; i++) {
            for (int j = 0; j < Matrix_1[0].length; j++) {
                Result[i][j] = Matrix_1[i][j].multiply(Scaler);
            }
        }
        return Result;
    }

    public static List<Eigen> PowerMethod(Double[][] Matrix, Double Tolerance, Integer k) {
        BigDecimal Matrix_copy[][] = new BigDecimal[Matrix.length][Matrix[0].length];
        for (int i = 0; i < Matrix_copy.length; i++) {
            for (int j = 0; j < Matrix_copy[i].length; j++) {
                Matrix_copy[i][j] = BigDecimal.valueOf(Matrix[i][j]).setScale(Scale, Mode);
            }
        }

        BigDecimal InitialGuess[][] = new BigDecimal[Matrix[0].length][1];
        int NoEigen = k;
        List<Eigen> EigenValues = new ArrayList<>();
        BigDecimal B[][] = Matrix_copy;
        int total_steps = 0;
        BigDecimal Tolerance_B = BigDecimal.valueOf(Tolerance).setScale(Scale, Mode);

        while (NoEigen > 0 && NoEigen <= Matrix.length) {
            for (int i = 0; i < Matrix.length; i++) {
                InitialGuess[i][0] = BigDecimal.valueOf(1.0).setScale(Scale, Mode);
            }
            BigDecimal initial = BigDecimal.valueOf(0.0).setScale(Scale, Mode);
            BigDecimal App_EigenValue = BigDecimal.valueOf(0.0).setScale(Scale, Mode);
            BigDecimal Error = BigDecimal.ONE.setScale(Scale, Mode);
            BigDecimal Norm = BigDecimal.valueOf(0.0).setScale(Scale, Mode);
            while (Error.compareTo(Tolerance_B) > 0) {
                total_steps++;
                Norm = BigDecimal.valueOf(0.0).setScale(Scale, Mode);
                BigDecimal eigenVector[][] = Matrix_Multiplication(B, InitialGuess);
                BigDecimal potential_EigenValue = BigDecimal.valueOf(0.0).setScale(Scale, Mode);
                for (int i = 0; i < eigenVector.length; i++) {
                    if (potential_EigenValue.abs().compareTo(eigenVector[i][0].abs()) < 0) {
                        potential_EigenValue = eigenVector[i][0];
                    }
                }
                App_EigenValue = potential_EigenValue;
                for (int i = 0; i < eigenVector.length; i++) {
                    InitialGuess[i][0] = eigenVector[i][0].divide(potential_EigenValue, Scale, Mode);
                    Norm = Norm.add(InitialGuess[i][0].pow(2));
                }
                if (App_EigenValue.compareTo(BigDecimal.ZERO) != 0) {
                    BigDecimal numerator = App_EigenValue.subtract(initial).abs();
                    BigDecimal denominator = App_EigenValue.abs();
                    Error = numerator.divide(denominator, Scale, Mode);
                } else {
                    Error = BigDecimal.ZERO.setScale(Scale, Mode);
                }
                initial = App_EigenValue;
            }
            String Array = "[";
            for (int i = 0; i < InitialGuess.length; i++) {
                Array += InitialGuess[i][0] + ", ";
            }
            Array += "]";
            // System.out.println(Array);
            Norm = Norm.sqrt(new MathContext(Scale));
            for (int i = 0; i < InitialGuess.length; i++) {
                InitialGuess[i][0] = InitialGuess[i][0].divide(Norm, Scale, Mode);
            }
            BigDecimal Direction[][] = Scaler(Matrix_Multiplication(InitialGuess, Transpose(InitialGuess)),
                    App_EigenValue);
            B = Matrix_Subtraction(B, Direction);
            // Eigen Eigen = new Eigen(App_EigenValue, InitialGuess);
            EigenValues.add(new Eigen(App_EigenValue, Array));
            NoEigen--;
        }
        System.out.println(total_steps);
        return EigenValues;
    }

    public static class Eigen {
        private BigDecimal EigenValue;
        private String EigenVector;

        public Eigen(BigDecimal EigenValue, String EigenVector) {
            this.EigenValue = EigenValue;
            this.EigenVector = EigenVector;
        }

        @Override
        public String toString() {
            String Expression = "{EigenValue: " + this.EigenValue;
            Expression += "\nEigenVector: " + this.EigenVector + "\n\n";
            return Expression;
        }
    }

    public static BigDecimal[] Jacobi_GaussIteration(Double[][] Matrix, Double[] EquationResult, Double Tolerance,
            Boolean Gauss) {
        BigDecimal Tolerance_B = BigDecimal.valueOf(Tolerance);
        BigDecimal Matrix_copy[][] = new BigDecimal[Matrix.length][Matrix[0].length];
        for (int i = 0; i < Matrix_copy.length; i++) {
            for (int j = 0; j < Matrix_copy[i].length; j++) {
                Matrix_copy[i][j] = BigDecimal.valueOf(Matrix[i][j]).setScale(Scale, Mode);
            }
        }

        BigDecimal[] EquationResult_copy = new BigDecimal[EquationResult.length];
        for (int i = 0; i < EquationResult.length; i++) {
            EquationResult_copy[i] = BigDecimal.valueOf(EquationResult[i]).setScale(Scale, Mode);
        }

        BigDecimal Sol[] = new BigDecimal[Matrix[0].length];
        if (EquationResult.length == Matrix.length) {
            BigDecimal Max_Error = BigDecimal.valueOf(1000000000);
            BigDecimal IterationSol[] = new BigDecimal[Matrix[0].length];
            for (int i = 0; i < IterationSol.length; i++) {
                IterationSol[i] = BigDecimal.valueOf(1.0).setScale(Scale, Mode);
                Sol[i] = BigDecimal.valueOf(1.0).setScale(Scale, Mode);
            }
            while (Max_Error.compareTo(Tolerance_B) == 1) {
                BigDecimal Error = BigDecimal.valueOf(0.0).setScale(Scale, Mode);
                for (int i = 0; i < Matrix_copy.length; i++) {
                    BigDecimal Sum = EquationResult_copy[i];
                    for (int j = 0; j < Matrix_copy[0].length; j++) {
                        if (j != i) {
                            Sum = Sum.subtract(Sol[j].multiply(Matrix_copy[i][j])).setScale(Scale, Mode);
                        }
                    }
                    IterationSol[i] = Sum.divide(Matrix_copy[i][i], Scale, Mode);
                    BigDecimal diff = IterationSol[i].subtract(Sol[i]).abs();
                    BigDecimal denominator = IterationSol[i].abs();
                    BigDecimal relativeError = diff.divide(denominator, Scale, Mode);
                    Error = Error.max(relativeError);
                    if (Double.isInfinite(Error.doubleValue())) {
                        Error = BigDecimal.valueOf(1.0).setScale(Scale, Mode);
                    }
                    if (Gauss == true)
                        Sol[i] = IterationSol[i];
                }
                Max_Error = Max_Error.min(Error);
                if (Gauss == false) {
                    for (int i = 0; i < Sol.length; i++) {
                        Sol[i] = IterationSol[i];
                    }
                }
            }
        } else {
            throw new Error();
        }
        return Sol;
    }
}
