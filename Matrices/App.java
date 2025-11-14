package Matrices;


public class App {
    public static void main(String[] args) {
        Double[][] Matrix = {
            {5.0,-1.0,1.0},
            {2.0,8.0,-1.0},
            {-1.0,1.0,3.0},
        };
        Double[] EquationResult = {10.0,10.0,10.0};
        Double Sol[] = MatrixFunction.Jacobi_GaussIteration(Matrix, EquationResult, 1e-9,true);
        for(int i =0;i<Sol.length;i++){
            System.out.println(Sol[i]);
        }

    }
}