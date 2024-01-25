---
layout: page
title: "PHYS 220: Intro to Computational Physics"
permalink: /UAF/CompPhys/Lecture-Extra
---

# [UAF](../../UAF.md): [Intro to Comp. Phys](CompPhys.md)
[Spring 2020] Intro to Computational Physics with Dr. Peter Delamere at UAF

---

# Lecture Extras

Some code from Dr. Delamere to supplement the lectures

*(For the last 6 notesbooks, I did't have the original code as files, but I re-created myself from scratch looking at the pdf version of it in the combined lecture notes.)

## [Cooling](Lecture-Extra/Cooling.html)

- Math & code examples for Newton's law of cooling

## [Array Syntax](Lecture-Extra/Array_syntax.html)

- Mainly just some examples and "tutorials" to learn more about Python's array syntax

## [Random Numbers](Lecture-Extra/Random_numbers.html)

- Notebook example to learn more about random number generation in Python

## [2D Monte Carlo](Lecture-Extra/2D_Monte_Carlo.html)

- Example for Multidimensional Monte Carlo Integration

## [Fourier Transform](Lecture-Extra/Fourier_Transform.html)*

- Mathematics of Fourier Transform
- Code example of use of *Fast Fourier Transform* (FFT)
    - `numpy.fft.rfft`

## [Uncertainty Principle](Lecture-Extra/Uncertainty_principle.html)*

- Example with using FFT for wavenumber *k* spectra of wave packets

## [Eigenvalues and eigenvectors](Lecture-Extra/Eigen.html)*

- Sample use of `numpy.linalg` module to calculate eigenvalues and eigenvectors
    - `eigh`
    - `eigvalsh`
- Explanation and example with *Asymmetric quantum well*
- Orthogonality of eigenvectors
- Symmetric potential well solution

## [Molecular Dynamics](Lecture-Extra/Molecular_dynamics.html)*

- Lennar-Jones potential
- (Example to simulate molecular dynamics I guess?)
- (Professor even incorporated code to animate the dynamics, but I haven't gotten it to work yet (especially in Quarto))

## [Gaussian Quadrature](Lecture-Extra/Gaussian.html)*

- Higher-order integration methods
    - trapezoidal rule
    - simpson's rule
- Gaussian Quadrature
    - Mathematics
    - Example code

## [Systems of Equations](Lecture-Extra/Systems_of_equations.html)*

- Solution of simultaneous linear equations
- Gaussian elimination
    - (an alternative to determining the inverse matrix)
    - Explanation and code example