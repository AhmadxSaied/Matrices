package Matrices;

import java.math.BigDecimal;
import java.util.List;

import Matrices.MatrixFunction_BigDecimal.Eigen;

public class App {
    public static void main(String[] args) {
        Double[][] Matrix = {
            {40.0,-20.0,0.0},
            {-20.0,40.0,-20.0},
            {0.0,-20.0,40.0},
        };
        Double[] result = {10.0,19.0,7.0};
        List<MatrixFunction_BigDecimal.Eigen> LU = MatrixFunction_BigDecimal.PowerMethod(Matrix,1e-18,3);
        System.out.println(LU);

    }
}