package Matrices;


public class App {
    public static void main(String[] args) {
        Double Matrix[][] = {
            {10.0,-7.0,0.0,7.0},
            {-3.0,2.099,6.0,3.901},
            {5.0,-1.0,5.0,6.0}
        };
        Double SolMat[][] = MatrixFunction.naive_Gauss_With_Pivot_and_Scale(Matrix);
        for(int i=0;i<SolMat.length;i++){
            for(int j=0;j<SolMat[0].length;j++){
                System.out.print(SolMat[i][j]+" ");
            }
            System.out.println();
        }
        Double Sol[] = MatrixFunction.BackwardSubstitution(Matrix);
        for(int i=0;i<Sol.length;i++){
            System.out.print(Sol[i]+" ");
        }
    }
}