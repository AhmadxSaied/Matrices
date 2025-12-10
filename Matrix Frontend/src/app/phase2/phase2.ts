import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';

interface Step {
  stepNumber: number;
  description: string;
  X_L?: number;
  X_U?: number;
  X_r?: number;
  F_Xl?: number;
  F_Xu?: number;
  F_Xr?: number;
  Xi_0?: number;
  Xi_1?: number;
  Error?: number;
}

interface RootResponse {
  status: string;
  result: number;
  executionTime: number;
  TotalIterations: number;
  steps: Step[];
  errorMessage: string | null;
}

@Component({
  selector: 'app-phase2',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './phase2.html',
  styleUrls: ['./phase2.css']
})
export class Phase2 {
  private http = inject(HttpClient);
  private router = inject(Router);

  functionStr: string = '';
  selectedMethod: string = 'Bisection';
  precision: number = 10;
  tolerance: number = 0.00001;
  maxIterations: number = 50;

  xl: number | null = null;
  xu: number | null = null;
  x0: number | null = null;
  x1: number | null = null;

  solutionData: RootResponse | null = null;
  isSolving: boolean = false;
  isPlotting: boolean = false;
  plotImage: string | null = null;
  showTable: boolean = false;

  methods = [
    { id: 'Bisection', name: 'Bisection Method' },
    { id: 'False_Position', name: 'False Position' },
    { id: 'FixedPoint', name: 'Fixed Point' },
    { id: 'Newton_Normal', name: 'Newton-Raphson' },
    { id: 'Newton_modified', name: 'Modified Newton' },
    { id: 'Secant', name: 'Secant Method' }
  ];

  goHome() {
    this.router.navigate(['/']);
  }

  solve() {
    if (!this.functionStr) return;

    this.isSolving = true;
    this.solutionData = null;
    this.showTable = false;

    const payload = {
      Function: this.functionStr,
      MethodId: this.selectedMethod,
      X_Lower: this.xl,
      X_Upper: this.xu,
      Xo_Initial: this.x0,
      X1_Initial: this.x1,
      Tolerance: this.tolerance,
      max_itr: this.maxIterations,
      percision: this.precision
    };

    this.http.post<RootResponse>('http://127.0.0.1:8000/solve_root', payload)
      .subscribe({
        next: (res) => {
          this.solutionData = res;
          this.isSolving = false;
          setTimeout(() => {
            const element = document.getElementById('results-section');
            if (element) element.scrollIntoView({ behavior: 'smooth' });
          }, 100);
        },
        error: (err) => {
          console.error(err);
          this.isSolving = false;
          alert('Backend connection failed or Solver Error');
        }
      });
  }

  plotFunction() {
    if (!this.functionStr) return;
    this.isPlotting = true;

    const payload = {
      equation: this.functionStr,
      start: (this.xl ?? (this.x0 ? this.x0 - 5 : -10)),
      end: (this.xu ?? (this.x0 ? this.x0 + 5 : 10))
    };

    this.http.post<any>('http://127.0.0.1:8000/plot', payload).subscribe({
      next: (res) => {
        this.plotImage = res.imageBase64;
        this.isPlotting = false;
      },
      error: () => {
        this.isPlotting = false;
        alert("Plotting service unavailable");
      }
    });
  }

  toggleTable() {
    this.showTable = !this.showTable;
    if (this.showTable) {
      setTimeout(() => {
        document.getElementById('steps-table')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }
  }

  get isBracketMethod(): boolean {
    return this.selectedMethod === 'Bisection' || this.selectedMethod === 'False_Position';
  }

  get isOpenMethod(): boolean {
    return this.selectedMethod === 'Newton_Normal' || this.selectedMethod === 'Newton_modified' || this.selectedMethod === 'FixedPoint';
  }

  get isSecant(): boolean {
    return this.selectedMethod === 'Secant';
  }
}