from pymimkl import (
    induce_linear_kernel, induce_polynomial_kernel, induce_gaussian_kernel,
    induce_sigmoidal_kernel
)
INDUCTION_FACTORY = {
    'induce_linear_kernel': induce_linear_kernel,
    'induce_polynomial_kernel': induce_polynomial_kernel,
    'induce_gaussian_kernel': induce_gaussian_kernel,
    'induce_sigmoidal_kernel': induce_sigmoidal_kernel,
}
