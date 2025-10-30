package Matrices;
import java.awt.BorderLayout;
import java.awt.Color;

import javax.swing.*;

public class App {
    public static void main(String[] args) {
        Double Matrix[][] = {
            {25.0,5.0,1.0},
            {64.0,8.0,1.0},
            {144.0,12.0,1.0},
        };
        Double SolMat[][][] = MatrixFunction.LU(Matrix);
        for(int i=0;i<SolMat[0].length;i++){
            for(int j=0;j<SolMat[0][0].length;j++){
                System.out.print(SolMat[0][i][j]+" ");
            }
            System.out.println();
        }
        System.out.println();
        for(int i=0;i<SolMat[1].length;i++){
            for(int j=0;j<SolMat[1][0].length;j++){
                System.out.print(SolMat[1][i][j]+" ");
            }
            System.out.println();
        }
        Double Y[] = MatrixFunction.LU_Solve(SolMat[0], SolMat[1], new Double[]{106.8,177.2,279.2});
        for(int i=0;i<Y.length;i++){
            System.out.println(Y[i]);
        }

    }
}