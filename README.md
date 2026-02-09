# 🧮 Numerical Computing Suite

A comprehensive GUI-based numerical analysis application developed for the Computer and Systems Engineering Department – Alexandria University.
The project implements and compares various numerical methods for solving linear systems and non-linear equations with high precision.

## 🚀 Project Overview

This repository contains a two-phase numerical solver:

Phase 1: Linear Systems

Focuses on solving systems of linear equations where the number of equations equals the number of unknowns.

Phase 2: Non-Linear Equations

Extends the solver to handle non-linear equations using root-finding algorithms.

## 🛠️ Key Features
### 🔹 Linear System Solvers (Phase 1)

Designed to solve systems with equal numbers of equations and variables.

* Direct Methods

* Gauss Elimination

* Gauss-Jordan

* LU Decomposition

* Doolittle

* Crout

* Cholesky

* Iterative Methods

* Jacobi Iteration

* Gauss-Seidel

* Additional Capabilities

* Customizable initial guesses

Stopping conditions:

* Maximum number of iterations

* Absolute relative error

* Automatic Partial Pivoting for improved numerical stability

### 🔹 Root Finders (Phase 2)

Handles non-linear equations involving:

Polynomials

Exponential functions

Trigonometric functions (sin, cos)

Implemented Methods

Bisection Method

False Position (Regula Falsi)

Fixed Point Iteration

Newton-Raphson

Original

Modified

Secant Method

Extra Features

Integrated function plotting to assist with initial guess selection

Displays execution run time for each method

## ✨ Advanced Specifications

Single-Step Simulation
Visualizes each iteration step-by-step for educational purposes.

Flexible Precision Control

User-defined significant figures

Adjustable epsilon (ε = 0.00001 by default)

Maximum iteration limit

Bullet-proof validation
Robust input validation for all supported equation formats.

## 💻 Tech Stack & Design

Programming Paradigm: Object-Oriented Programming (OOP)

Architecture: Modular, scalable, and clean design

GUI: Interactive interface with:

Drop-down menus for method selection

Clear result visualization

Analysis: Comparison of methods based on:

Convergence behavior

Approximate error

Execution time

## 👥 Team

This project was completed by a team of 5 Computer and Systems Engineering students, collaborating across both development phases.

## 📜 Academic Context

Institution: Alexandria University

Faculty: Faculty of Engineering

Department: Computer and Systems Engineering

Course: Numerical Computing

Academic Year: 2023–2024
