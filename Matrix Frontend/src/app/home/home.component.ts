import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="solver-container" style="text-align: center; min-height: 80vh; display: flex; flex-direction: column; justify-content: center; align-items: center;">
      <h1 style="font-size: 3rem; margin-bottom: 50px;">Numerical Analysis Solver</h1>
      <div style="display: flex; gap: 40px; justify-content: center; flex-wrap: wrap;">
        <div class="phase-card" (click)="nav('phase1')">
          <div class="icon">∑</div>
          <h2>Phase 1</h2>
          <p>Matrix Solver</p>
        </div>
        <div class="phase-card" (click)="nav('phase2')">
          <div class="icon">ƒ(x)</div>
          <h2>Phase 2</h2>
          <p>Root Finder</p>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .phase-card {
      background: rgba(30, 30, 63, 0.6); border: 1px solid rgba(255,255,255,0.1);
      padding: 50px; border-radius: 25px; cursor: pointer; transition: all 0.3s ease; width: 300px;
    }
    .phase-card:hover { transform: translateY(-10px); border-color: #00e0ff; box-shadow: 0 0 30px rgba(0, 224, 255, 0.2); }
    .icon { font-size: 5rem; color: #ff4b91; margin-bottom: 20px; }
    h2 { color: #fff; margin: 10px 0; }
    p { color: #a0aec0; }
  `]
})
export class HomeComponent {
  constructor(private router: Router) {}
  nav(path: string) { this.router.navigate([path]); }
}