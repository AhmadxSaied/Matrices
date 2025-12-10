import { Component, OnInit } from '@angular/core';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  templateUrl: './app.html',
  styleUrls: ['./app.css']
})
export class App implements OnInit {
  isDarkMode: boolean = true;

  ngOnInit() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') {
      this.setLightMode();
    } else {
      this.setDarkMode();
    }
  }

  toggleTheme() {
    this.isDarkMode = !this.isDarkMode;
    if (this.isDarkMode) {
      this.setDarkMode();
    } else {
      this.setLightMode();
    }
  }

  private setDarkMode() {
    this.isDarkMode = true;
    document.body.removeAttribute('data-theme');
    localStorage.setItem('theme', 'dark');
  }

  private setLightMode() {
    this.isDarkMode = false;
    document.body.setAttribute('data-theme', 'light');
    localStorage.setItem('theme', 'light');
  }
}