import { Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { Phase1 } from './phase1/phase1'; 
import { Phase2 } from './phase2/phase2'; 

export const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'phase1', component: Phase1 },
  { path: 'phase2', component: Phase2 },
  { path: '**', redirectTo: '' }
];