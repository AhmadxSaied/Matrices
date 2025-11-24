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
      steps: [
        {
            stepNumber: 1,
            description: "select pivot",
            currentMatrixA: [[2,1,4],[1,2,3],[4,-1,2]],
            currentMatrixB : [1,1.5,2],
            pivotIndex: { r: 0, c: 0 },
          },
        {
            stepNumber: 2,
            description: "-0.5*r1+r2 ===> r2",
            currentMatrixA: [[2,1,4],[0,1.5,1],[4,-1,2]],
            currentMatrixB : [1,1,2],
            pivotIndex: { r: 0, c: 0 },
            highlightRow : 1
        },
        {
            stepNumber: 3,
            description: "-2*r1+r3 ===> r3",
            currentMatrixA: [[2,1,4],[0,1.5,1],[0,-3,-6]],
            currentMatrixB : [1,1,0],
            pivotIndex: { r: 0, c: 0 },
            highlightRow : 2
        },
        {
            stepNumber: 4,
            description: "select pivot",
            currentMatrixA: [[2,1,4],[0,1.5,1],[0,-3,-6]],
            currentMatrixB : [1,1,0],
            pivotIndex: { r: 1, c: 1 },
        },
        {
            stepNumber: 5,
            description: "2*r2+r3 ===> r3",
            currentMatrixA: [[2,1,4],[0,1.5,1],[0,0,-4]],
            currentMatrixB : [1,1,2],
            pivotIndex: { r: 1, c: 1 },
            highlightRow : 2
        },
        {
          stepNumber:6,
          description:"x3 = 2/-4 = -0.5",
          currentMatrixA:[[2,1,4],[0,1.5,1],[0,0,1]],
          currentMatrixB : [1,1,-0.5],
        },
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