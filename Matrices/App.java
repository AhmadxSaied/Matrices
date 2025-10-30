package Matrices;


public class App {
    public static void main(String[] args) {
        Double[][] Matrix = {
            {9.83, 2.41, 1.92, 3.56, 2.88},
            {2.41, 7.65, 2.37, 1.84, 2.01},
            {1.92, 2.37, 8.49, 2.62, 1.95},
            {3.56, 1.84, 2.62, 9.27, 2.73},
            {2.88, 2.01, 1.95, 2.73, 8.94}
        };
        Double SolMat[][] = MatrixFunction.Cholesky(Matrix);
        for(int i=0;i<SolMat.length;i++){
            for(int j=0;j<SolMat[0].length;j++){
                System.out.print(SolMat[i][j]+" ");
            }
            System.out.println();
        }
        System.out.println();
        // for(int i=0;i<SolMat[1].length;i++){
        //     for(int j=0;j<SolMat[1][0].length;j++){
        //         System.out.print(SolMat[1][i][j]+" ");
        //     }
        //     System.out.println();
        // }
        // Double Y[] = MatrixFunction.LU_Solve(SolMat[0], SolMat[1], new Double[]{3.0,3.0,9.0});
        // for(int i=0;i<Y.length;i++){
        //     System.out.println(Y[i]);
        // }

    }
}