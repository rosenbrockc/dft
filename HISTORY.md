# Revision History

## Revision 0.0.7

- Added the electron-electron interaction, but not yet exchange correlation.
- H is no longer hermitian with the new part.

## Revision 0.0.6

- Complete wave function solution.
- Added steepest descent algorithm.
- All unit tested (with at least one test using random vectors and matrices).

## Revision 0.0.5

- Added energy solver for arbitrary wave functions.

## Revision 0.0.4

- Completed up to section 3.3, `diagouter` function. All unit-tested and passing.

## Revision 0.0.3

- Added `scipy` as requirement.

## Revision 0.0.2

- Working Ewald solver that produces good values. However, it is quite sensitive to choice of `R`.
- Unit tests for Arias quick and dirty solver; it doesn't reproduce the values he suggests in the assigment.
- Trigger CI and coverage automation.

## Revision 0.0.1

- Completed the poisson solver using FFT.
- Unit tests complete for poisson.