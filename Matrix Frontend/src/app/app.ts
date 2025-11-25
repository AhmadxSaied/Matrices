import { Component, ViewEncapsulation } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

interface MatrixStep {
  stepNumber: number;
  description: string;
  currentMatrixA: number[][];
  currentMatrixB : number[];
  pivotIndex?: { r: number, c: number }; 
  highlightRow?: number;
  L?:number[][];
  U?:number[][];
}

interface SolutionResponse {
  status: string;
  results: number[];
  executionTime: number ;
  totalIterations : number;
  steps: MatrixStep[];
  errorMessage : null
  equations?: string[];
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrls: ['./app.css'], 
  encapsulation : ViewEncapsulation.None 
})
export class App {
  //Inputs
  NumberEquations: number | null = null;
  
  //Data Structures
  matrixA: number[][] = [];
  matrixB: number[] = [];
  
  //UI State
  isGenerated: boolean = false;
  isSolved: boolean = false;
  showError : boolean = false ;
  //Method Selection
  selectedMethod: string = 'GAUSS';
  methods = [
    { id: 'GAUSS', name: 'Gauss Elimination' },
    { id: 'JORDAN', name: 'Gauss-Jordan' },
    { id: 'LU', name: 'LU Decomposition' },
    { id: 'JACOBI', name: 'Jacobi Iteration' },
    { id: 'SEIDEL', name: 'Gauss-Seidel' }
  ];

  //Method Parameters
  luForm: string = 'DOOLITTLE';
  initialGuess = new Array(this.NumberEquations).fill(0);
  maxIterations: number = 50;
  tolerance: number = 0.0001;

  //Solution Player State
  solutionData: SolutionResponse | null = null;
  currentStepIndex: number = 0;
  currentMatrixADisplay: number[][] = [];
  currentMatrixBDisplay: number[] = [];
  progressPercentage :number = 0;
  // 1. Generate Grid (N x N+1)
  gridGenerator() {
    if (!this.NumberEquations || this.NumberEquations < 1) {

      alert("Please enter a valid dimension (N >= 1)");
      return;
    }

    this.matrixA = [];
    this.matrixB = [];
    this.initialGuess = [];
    this.showError = false ;

    for (let i = 0; i < this.NumberEquations; i++) {
      const row = new Array(this.NumberEquations).fill(null);
      this.matrixA.push(row);
      this.matrixB.push(null as any);
      this.initialGuess.push(0);
    }

    this.isGenerated = true;
    this.isSolved = false;
  }
  isValid() : boolean{
    for(let i = 0 ; i < this.matrixA.length ; i++){
      for(let j = 0 ; j < this.matrixB.length ; j++){
        if(this.matrixA[i][j]==null || this.matrixA[i][j] == undefined )return false ;

      }
      if(this.matrixB[i]==null || this.matrixB[i] == undefined )return false ;

    }
    return true ;
  }

  // 2. Solve & Create Mock Response
  solve() {

    this.showError = true ;

    if(!this.isValid()){
      this.animateInvalidCells();
      return ;
    }
    this.progressPercentage = 0;
    const payload = {
      matrixA: this.matrixA,
      matrixB: this.matrixB,
      methodID : this.selectedMethod,
        methodParams: {
          luForm: this.selectedMethod === 'LU' ? this.luForm : null,
          initialGuess: (this.selectedMethod === 'JACOBI' || this.selectedMethod === 'SEIDEL') ? this.initialGuess : null,
          maxIterations: this.maxIterations,
          tolerance: this.tolerance
        }
    };

    console.log("Sending Payload to Backend:", payload);

    this.solutionData = {
      status: "SUCCESS",
      results: [2, 5, 6],
      executionTime: 5 ,
      totalIterations : 6,
      equations:["x1 = x2/3","x2 = x3/3","x3 = x1/x2"],
      steps: [{
        stepNumber: 1,
        description: "diagonally dominant -> True",
        currentMatrixA: [
          [4, 2, 2],
          [2, 5, 3],
          [2, 3, 6]
        ],
        currentMatrixB: [8, 10, 11],
      },
        {
          stepNumber: 2,
          description: "Iter 1: Vector=[2.0000, 1.2000, 0.5667] | Error=1.0000",
          currentMatrixA: [
            [4, 2, 2],
            [2, 5, 3],
            [2, 3, 6]
          ],
          currentMatrixB: [2.0000000000, 1.2000000000, 0.5666666667],
        },
        {
          stepNumber: 3,
          description: "Iter 2: Vector=[1.1167, 1.2133, 0.8544] | Error=0.7910",
          currentMatrixA: [
            [4, 2, 2],
            [2, 5, 3],
            [2, 3, 6]
          ],
          currentMatrixB: [1.1166666670, 1.2133333330, 0.8544444445],
        },
        {
          stepNumber: 4,
          description: "Iter 3: Vector=[0.9661, 1.1009, 0.9609] | Error=0.1558",
          currentMatrixA: [
            [4, 2, 2],
            [2, 5, 3],
            [2, 3, 6]
          ],
          currentMatrixB: [0.9661111112, 1.1008888890, 0.9608518518],
        },
        {
          stepNumber: 5,
          description: "Iter 4: Vector=[0.9691, 1.0358, 0.9924] | Error=0.0628",
          currentMatrixA: [
            [4, 2, 2],
            [2, 5, 3],
            [2, 3, 6]
          ],
          currentMatrixB: [0.9691296295, 1.0358370370, 0.9923716050],
        },
        {
          stepNumber: 6,
          description: "Iter 5: Vector=[0.9859, 1.0102, 0.9996] | Error=0.0254",
          currentMatrixA: [
            [4, 2, 2],
            [2, 5, 3],
            [2, 3, 6]
          ],
          currentMatrixB: [0.9858956790, 1.0102187650, 0.9995920578],
        },
        {
          stepNumber: 7,
          description: "Iter 6: Vector=[0.9951, 1.0022, 1.0005] | Error=0.0092",
          currentMatrixA: [
            [4, 2, 2],
            [2, 5, 3],
            [2, 3, 6]
          ],
          currentMatrixB: [0.9950945885, 1.0022069300, 1.0005316720],
        },
        {
          stepNumber: 8,
          description: "Iter 7: Vector=[0.9986, 1.0002, 1.0003] | Error=0.0035",
          currentMatrixA: [
            [4, 2, 2],
            [2, 5, 3],
            [2, 3, 6]
          ],
          currentMatrixB: [0.9986306990, 1.0002287170, 1.0003420750],
        },
        {
          stepNumber: 9,
          description: "Iter 8: Vector=[0.9997, 0.9999, 1.0001] | Error=0.0011",
          currentMatrixA: [
            [4, 2, 2],
            [2, 5, 3],
            [2, 3, 6]
          ],
          currentMatrixB: [0.9997146040, 0.9999089134, 1.0001406750],
        },
        {
          stepNumber: 10,
          description: "Iter 9: Vector=[1.0000, 1.0000, 1.0000] | Error=0.0003",
          currentMatrixA: [
            [4, 2, 2],
            [2, 5, 3],
            [2, 3, 6]
          ],
          currentMatrixB: [0.9999752058, 0.9999255126, 1.0000455080],
        },
        {
          stepNumber: 11,
          description: "Iter 10: Vector=[1.0000, 1.0000, 1.0000] | Error=0.0000",
          currentMatrixA: [
            [4, 2, 2],
            [2, 5, 3],
            [2, 3, 6]
          ],
          currentMatrixB: [1.0000144900, 0.9999668992, 1.0000117200],
        },
        {
          stepNumber: 12,
          description: "Iter 11: Vector=[1.0000, 1.0000, 1.0000] | Error=0.0000",
          currentMatrixA: [
            [4, 2, 2],
            [2, 5, 3],
            [2, 3, 6]
          ],
          currentMatrixB: [1.0000106900, 0.9999886920, 1.0000020910],
        },
        {
          stepNumber: 13,
          description: "Iter 12: Vector=[1.0000, 1.0000, 1.0000] | Error=0.0000",
          currentMatrixA: [
            [4, 2, 2],
            [2, 5, 3],
            [2, 3, 6]
          ],
          currentMatrixB: [1.0000046080, 0.9999969022, 1.0000000130],
        }
      ],
      errorMessage:null
    };
    this.isSolved = true;
    this.currentStepIndex = 0;
    this.updateDisplayMatrix();
  }


  animateInvalidCells() {
    setTimeout(() => {
      const cells = document.querySelectorAll<HTMLElement>('.puzzle-cell.invalid');
      cells.forEach(cell => {
        cell.classList.remove('animate');
        void (cell.offsetWidth);
        cell.classList.add('animate');
      });
    }, 0);
  }

  //Solution Player Logic

  nextStep() {
    if (this.solutionData && this.currentStepIndex < this.solutionData.steps.length - 1) {
      this.currentStepIndex++;
      this.updateDisplayMatrix();
      this.getPercentage();

    }
  }

  prevStep() {
    if (this.currentStepIndex > 0) {
      this.currentStepIndex--;
      this.updateDisplayMatrix();
      this.getPercentage();
    }
  }

  updateDisplayMatrix() {
    if (this.solutionData) {
      this.currentMatrixADisplay = this.solutionData.steps[this.currentStepIndex].currentMatrixA;
      this.currentMatrixBDisplay = this.solutionData.steps[this.currentStepIndex].currentMatrixB;

      
    }
  }

  getCellStyle(rowIndex: number, colIndex: number) {
    if (!this.solutionData) return {};

    const step = this.solutionData.steps[this.currentStepIndex];
    const aCols = this.currentMatrixADisplay && this.currentMatrixADisplay[0] ? this.currentMatrixADisplay[0].length : 0;
    const isBCol = colIndex === aCols;

    // Highlight pivot cell
    if (step.pivotIndex && step.pivotIndex.r === rowIndex && step.pivotIndex.c === colIndex) {
      return {
        'border-color': 'var(--success-color)',
        'box-shadow': '0 0 15px var(--success-color)',
        'transform': 'scale(1.05)'
      };
    }

    // Highlight entire row

    if (step.highlightRow === rowIndex && (colIndex < aCols || isBCol)) {
      if (isBCol) {
        return {
          'background': 'rgba(0, 224, 255, 0.18)' 
        };
      }

      return {
        'background': 'rgba(255, 75, 145, 0.2)'
      };
    }
    return {};
  }

  getPercentage(){
    if (!this.solutionData) return 0;
    this.progressPercentage = ((this.currentStepIndex) / (this.solutionData.steps.length-1)) * 100;
    return ;
  }
}