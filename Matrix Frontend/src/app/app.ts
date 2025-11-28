import { Component, ViewEncapsulation, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

interface MatrixStep {
  stepNumber: number;
  description: string;
  matrixA: number[][]; 
  matrixB : number[];
  L?: number[][];
  U?: number[][];
  pivotIndex?: { r: number, c: number }; 
  highlightRow?: number;
}

interface SolutionResponse {
  status: string;
  results: number[];
  executionTime: number ;
  TotalIternation : number; 
  steps: MatrixStep[];
  errorMessage : string | null;
  equations?: string[];
  L?: number[][];
  U?: number[][];
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
  private http = inject(HttpClient);

  NumberEquations: number | null = null;
  
  matrixA: number[][] = [];
  matrixB: number[] = [];
  precision: number = 10;
  isGenerated: boolean = false;
  isSolved: boolean = false;
  showError : boolean = false;
  showSteps: boolean = false; 
  withpivoting: boolean = false;
  withscaling: boolean = false;
  show:boolean = false;
  currentMatrixL: number[][] | null = null;
  currentMatrixU: number[][] | null = null;
  selectedMethod: string = 'Naive_Gauss';
  methods = [
    { id: 'Naive_Gauss', name: 'Naive Gauss Elimination' },
    {id:'Gauss_elimination',name: 'Gauss Elimination' },
    { id: 'Gauss_Jordan', name: 'Gauss-Jordan' },
    { id: 'LU', name: 'LU Decomposition' },
    { id: 'JACOBI', name: 'Jacobi Iteration' },
    { id: 'Gauss_Seidel', name: 'Gauss-Seidel' },
  ];

  luForm: string = 'DOOLITTLE';
  initialGuess: number[] = [];
  maxIterations: number = 50;
  tolerance: number = 0.0001;

  solutionData: SolutionResponse | null = null;
  currentStepIndex: number = 0;
  currentMatrixADisplay: number[][] = [];
  currentMatrixBDisplay: number[] = [];
  progressPercentage :number = 0;

  gridGenerator() {
    if (!this.NumberEquations || this.NumberEquations < 1) {
      alert("Please enter a valid dimension (N >= 1)");
      return;
    }
    this.matrixA = [];
    this.matrixB = [];
    this.initialGuess = [];
    this.showError = false;
    this.isSolved = false;
    this.showSteps = false;

    for (let i = 0; i < this.NumberEquations; i++) {
      const row = new Array(this.NumberEquations).fill(null);
      this.matrixA.push(row);
      this.matrixB.push(null as any);
      this.initialGuess.push(0);
    }
    this.isGenerated = true;
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

  solve() {
    this.showError = true;

    if(!this.isValid()){
      this.animateInvalidCells();
      return;
    }
    
    this.progressPercentage = 0;
    this.showSteps = false;

    let finalMethodID = this.selectedMethod;
    if (this.selectedMethod === 'LU') {
        if (this.luForm === 'DOOLITTLE') finalMethodID = 'LU_decomposition_Doolittle';
        if (this.luForm === 'CROUT') finalMethodID = 'LU_decomposition_Croud';
        if (this.luForm === 'CHOLESKY') finalMethodID = 'LU_decomposition_Cholesky';
    } else if (this.selectedMethod === 'JACOBI') {
        finalMethodID = 'Jacobi';
    } else if (this.selectedMethod === 'Gauss_Seidel') { 
        finalMethodID = 'Gauss_Seidel';
    }
     else if(this.selectedMethod === 'Gauss_Jordan') {
       finalMethodID = 'Gauss_Jordan';
    }
    else{
      if(this.withscaling){
        finalMethodID = 'Gauss_elimination_Pivoting_Scaling';
      }
      else if(this.withpivoting){
        finalMethodID ='Gauss_elimination_Pivoting';
      }
      else{
        finalMethodID ='Naive_Gauss';
      }
    }

    const payload = {
      MethodId: finalMethodID,
      precision: this.precision,
      size: this.NumberEquations,
      matrix: this.matrixA,
      vector_of_sol: this.matrixB,
      initial_guess: (this.selectedMethod === 'JACOBI' || this.selectedMethod === 'Gauss_Seidel') ? this.initialGuess : [],
      max_iterations: this.maxIterations,
      Tolerance: this.tolerance,
      methodParams: {} 
    };

    console.log("Sending Payload:", payload);

    this.http.post<SolutionResponse>('http://127.0.0.1:8000/solve', payload)
      .subscribe({
        next: (response) => {
          if(this.selectedMethod !== 'JACOBI' && this.selectedMethod !== 'Gauss_Seidel') {
          if (response.errorMessage) {
            alert("Solver Error: " + response.errorMessage);
            return;
          }
          }
          else{
            if (response.errorMessage === 'can\'t divide by zero') {
              alert("Solver Error: " + response.errorMessage);
              return;
            }
          }
          this.solutionData = response;
          this.isSolved = true;
          this.currentStepIndex = 0;
          this.updateDisplayMatrix();
          this.getPercentage();
        },
        error: (err) => {
          console.error("Connection Error:", err);
          alert("Failed to connect to backend. Is uvicorn running?");
        }
      });
  }

  viewSteps() { this.showSteps = true; }
  backToResults() { this.showSteps = false; }

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
      const step = this.solutionData.steps[this.currentStepIndex];
      this.currentMatrixADisplay = step.matrixA;
      this.currentMatrixBDisplay = step.matrixB;
      this.currentMatrixL = this.solutionData.L || null;
      this.currentMatrixU = this.solutionData.U || null;
    }
  }

  getCellStyle(rowIndex: number, colIndex: number) {
    if (!this.solutionData) return {};
    const step = this.solutionData.steps[this.currentStepIndex];
    
    if (step.pivotIndex && step.pivotIndex.r === rowIndex && step.pivotIndex.c === colIndex) {
        return {
          'border-color': 'var(--success-color)',
          'box-shadow': '0 0 15px var(--success-color)',
          'transform': 'scale(1.05)'
        };
      }
  
      const aCols = this.currentMatrixADisplay && this.currentMatrixADisplay[0] ? this.currentMatrixADisplay[0].length : 0;
      const isBCol = colIndex === aCols;

      if (step.highlightRow === rowIndex && (colIndex < aCols || isBCol)) {
        if (isBCol) {
          return { 'background': 'rgba(0, 224, 255, 0.18)' };
        }
        return { 'background': 'rgba(255, 75, 145, 0.2)' };
      }
    return {};
  }

getPercentage() {
    if (!this.solutionData || !this.solutionData.steps || this.solutionData.steps.length === 0) {
        this.progressPercentage = 0;
        return;
    }
    this.progressPercentage = ((this.currentStepIndex + 1) / this.solutionData.steps.length) * 100;
  }
}