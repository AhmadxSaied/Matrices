# Phase 2 API Documentation

This document outlines the API endpoints and data structures required for Phase 2 of the application.

## Endpoints

### 1. Solve Root
**URL:** `http://127.0.0.1:8000/solve_root`
**Method:** `POST`

#### Request Body
The backend expects a JSON object with the following structure:

```json
{
  "Function": "string",       // The mathematical function to solve (e.g., "x^2 - 4")
  "MethodId": "string",       // The ID of the method to use. Options:
                              // "Bisection", "False_Position", "FixedPoint",
                              // "Newton_Normal", "Newton_modified", "Secant"
  "X_Lower": number | null,   // Lower bound (required for bracket methods)
  "X_Upper": number | null,   // Upper bound (required for bracket methods)
  "Xo_Initial": number | null,// Initial guess (required for open methods)
  "X1_Initial": number | null,// Second initial guess (required for Secant)
  "Tolerance": number,        // Error tolerance (e.g., 0.00001)
  "max_itr": number,          // Maximum number of iterations (e.g., 50)
  "percision": number         // Number of decimal places for precision
}
```

#### Response Body
The backend returns a JSON object matching the `RootResponse` structure:

```json
{
  "status": "string",         // Status of the operation (e.g., "success", "error")
  "result": number,           // The calculated root
  "executionTime": number,    // Time taken to execute in seconds
  "TotalIterations": number,  // Total number of iterations performed
  "errorMessage": string | null, // Error message if any, otherwise null
  "steps": [                  // Array of steps detailing the iteration process
    {
      "stepNumber": number,
      "description": "string",
      "X_L": number,          // Optional: Lower bound at this step
      "X_U": number,          // Optional: Upper bound at this step
      "X_r": number,          // Optional: Root estimate at this step
      "F_Xl": number,         // Optional: Function value at lower bound
      "F_Xu": number,         // Optional: Function value at upper bound
      "F_Xr": number,         // Optional: Function value at root estimate
      "Xi_0": number,         // Optional: Previous approximation
      "Xi_1": number,         // Optional: Current approximation
      "Error": number         // Relative error at this step
    }
  ]
}
```

### 2. Plot Function
**URL:** `http://127.0.0.1:8000/plot`
**Method:** `POST`

#### Request Body
```json
{
  "equation": "string",       // The function equation to plot
  "start": number,            // Start x-value for the plot range
  "end": number               // End x-value for the plot range
}
```

#### Response Body
```json
{
  "imageBase64": "string"     // Base64 encoded string of the generated plot image
}
```
