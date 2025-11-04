package Matrices;

import java.util.List;

public class App {
    public static void main(String[] args) {
        Double[][] Matrix = {
            {2.0,6.0},
            {0.0,3.0}
        };
        List<Double> Sol = MatrixFunction.PowerMethod(Matrix,1e-9,Matrix.length);
        System.out.println(Sol);

    }
}