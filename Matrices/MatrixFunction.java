package Matrices;


public class MatrixFunction {
    /*
     * If there is any bugs please try to solve them and tell me
     */
    public static Double[][] naive_Gauss(Double Matrix[][]) {
        Double Matrix_copy[][] = Matrix;
        /*
         * Dont forget that naive gauss doesnt deal with division be zeros so we
         * must be prepared to catch the Division by zero error.
         */
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
                    double Multiplier = (Matrix_copy[j][i]) / (Matrix_copy[i][i]);
                    for (int k = 0; k < Matrix_copy[i].length; k++) {
                        Matrix_copy[j][k] = Matrix_copy[i][k] * Multiplier - Matrix_copy[j][k];
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

    public static Double[][] naive_Gauss_With_Pivot(Double Matrix[][]) {
        Double Matrix_copy[][] = Matrix;
        try {
            for (int i = 0; i < Matrix_copy.length; i++) {
                /*
                 * Here We want to get the Highest Value in The column we are currently in
                 * And Strore its Row index
                 * After that we want to swap the row we are currently pointing to
                 * to the Row With the highest magnitude pivot value
                 */
                Double Max = Double.NEGATIVE_INFINITY;
                int Max_index = -1;
                for (int j = i; j < Matrix_copy.length; j++) {
                    if (Math.abs(Matrix_copy[j][i]) > Max) {
                        Max = Math.abs(Matrix_copy[j][i]);
                        Max_index = j;
                    }
                }
                /*
                 * Here We assign a refrence to the row we are in
                 * And then we swap the row we are in with the row with the highest magnitude
                 * pivot We then replace the row we and in with the row we swapped
                 * And Voala we have introduced Pivoting
                 */
                Double Temp[] = Matrix_copy[i];
                Matrix_copy[i] = Matrix_copy[Max_index];
                Matrix_copy[Max_index] = Temp;
                /*
                 * Here we repeat what we discussed Above;
                 */
                for (int j = i + 1; j < Matrix_copy.length; j++) {
                    double Multiplier = (Matrix_copy[j][i]) / (Matrix_copy[i][i]);
                    for (int k = 0; k < Matrix_copy[i].length; k++) {
                        Matrix_copy[j][k] = (Matrix_copy[i][k] * Multiplier) - Matrix_copy[j][k];
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

    public static Double[][] naive_Gauss_With_Pivot_and_Scale(Double Matrix[][]) {
        Double Matrix_copy[][] = Matrix;
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
                double overAllMax = Double.NEGATIVE_INFINITY;
                double scale[] = new double[Matrix.length];
                for (int j = i; j < Matrix_copy.length; j++) {
                    double max = Double.NEGATIVE_INFINITY;
                    for (int k = j; k < Matrix_copy[0].length - 1; k++) {
                        if (Matrix_copy[j][k] > max) {
                            max = Matrix_copy[j][k];
                        }
                    }
                    /*
                     * Here We calculate the scale which is pivot/ max value in row 
                     */
                    scale[j] = Math.abs(Matrix_copy[j][j] / max);
                    if (scale[j] > overAllMax) {
                        overAllMax = scale[j];
                        Max_index = j;
                    }
                }
                Double Temp[] = Matrix_copy[i];
                Matrix_copy[i] = Matrix_copy[Max_index];
                Matrix_copy[Max_index] = Temp;
                /*
                 * Here is the same as before nothing new
                 */
                for (int j = i + 1; j < Matrix_copy.length; j++) {
                    double Multiplier = (Matrix_copy[j][i]) / (Matrix_copy[i][i]);
                    for (int k = 0; k < Matrix_copy[i].length; k++) {
                        Matrix_copy[j][k] = (Matrix_copy[i][k] * Multiplier) - Matrix_copy[j][k];
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
}