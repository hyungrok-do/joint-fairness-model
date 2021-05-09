
cimport cython

import numpy as np
cimport numpy as np
from libcpp cimport bool

from scipy.linalg.cython_blas cimport dgemv
from scipy.linalg.cython_blas cimport ddot
from scipy.linalg.cython_blas cimport dscal
from scipy.linalg.cython_blas cimport dcopy
np.import_array()

cdef extern from "math.h":
    double exp(double x)
    double sqrt(double x)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef logistic(double[:] x, int n):
    cdef double[:] z = x
    cdef int i
    for i in range(n):
        z[i] = 1 / (1 + exp(-1 * x[i]))
    return z

@cython.boundscheck(False)
@cython.wraparound(False)
cdef soft_thresholding(double x, double lam):
    if x > lam:
        return x - lam
    elif x < -lam:
        return x + lam
    else:
        return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef LinfBallProjection(double x):
    if x > 1:
        return 1
    elif x < -1:
        return -1
    else:
        return x

@cython.boundscheck(False)
@cython.wraparound(False)
cdef vecnorm(double[:] v, int n):
    cdef int i
    cdef double d = 0

    for i in range(n):
        d += v[i]*v[i]
    return sqrt(d)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef vecsub(double[:] v1, double[:] v2, int n):
    cdef int i
    cdef double[:] d = np.zeros(n)

    for i in range(n):
        d[i] = v1[i] - v2[i]
    return d

@cython.boundscheck(False)
@cython.wraparound(False)
cdef vecadd(double[:] v1, double[:] v2, int n):
    cdef int i
    cdef double[:] d = np.zeros(n)

    for i in range(n):
        d[i] = v1[i] + v2[i]
    return d

@cython.boundscheck(False)
@cython.wraparound(False)
def logistic_lasso_gradient_solver(double[:,:] x, double[:] y, double t, double lam, int maxiter, double tol):
    cdef int n = x.shape[1]
    cdef int p = x.shape[0]

    cdef double[:] beta = np.zeros(p)
    cdef double[:] _beta = np.zeros(p)
    cdef double[:] alpha = np.zeros(p)
    cdef double[:] gamma = np.zeros(p)
    cdef double[:] grad = np.zeros(p)
    cdef double[:] momentum = np.zeros(p)

    cdef double[:] s = np.ones(2)
    cdef double[:] xb = np.zeros(n)
    cdef double[:] err = np.zeros(n)
    cdef double[:] pr = np.zeros(n)

    cdef double double_zero = 0
    cdef int one = 1
    cdef double double_one = 1
    cdef double mm = 1

    cdef int i
    cdef int j

    for i in range(maxiter):
        # get xb = matmul(x, gamma)
        dgemv('T', &p, &n, &double_one, &x[0,0], &p, &gamma[0], &one, &double_zero, &xb[0], &one)

        # get grad = matmul(tr(x), pr - y)
        pr = logistic(xb, n)
        err = vecsub(pr, y, n)

        dgemv('N', &p, &n, &double_one, &x[0,0], &p, &err[0], &one, &double_zero, &grad[0], &one)
        dscal(&p, &t, &grad[0], &one)

        alpha = vecsub(gamma, grad, p)

        dcopy(&p, &beta[0], &one, &_beta[0], &one)
        dcopy(&p, &alpha[0], &one, &beta[0], &one)

        for j in range(1, p):
            beta[j] = soft_thresholding(alpha[j], lam)

        if vecnorm(vecsub(_beta, beta, p), p) < tol:
            break

        s[1] = (1 + sqrt(1 + 4 * s[0] * s[0])) * 0.5
        mm = (s[0] - 1) / s[1]

        momentum = vecsub(beta, _beta, p)
        dscal(&p, &mm, &momentum[0], &one)
        gamma = vecadd(beta, momentum, p)

        s[0] = s[1]

    return np.array(beta)



@cython.boundscheck(False)
@cython.wraparound(False)
def single_fair_logistic_solver(double[:,:] x, double[:] y, double[:,:] F, double t, double mu, double lam, int maxiter, double tol):
    cdef int n = x.shape[1]
    cdef int p = x.shape[0]
    cdef int m = F.shape[1]

    cdef double[:] beta = np.zeros(p)
    cdef double[:] _beta = np.zeros(p)
    cdef double[:] alpha = np.zeros(p)
    cdef double[:] gamma = np.zeros(p)
    cdef double[:] grad = np.zeros(p)
    cdef double[:] grad_F = np.zeros(p)
    cdef double[:] momentum = np.zeros(p)

    cdef double[:] a = np.zeros(p)
    cdef double[:] s = np.ones(2)
    cdef double[:] xb = np.zeros(n)
    cdef double[:] err = np.zeros(n)
    cdef double[:] pr = np.zeros(n)

    cdef double double_zero = 0
    cdef double double_one = 1
    cdef int one = 1
    cdef double mm = 1
    cdef double mu_inv = 1/mu
    cdef double n_inv = 1/np.float64(n)

    cdef int i
    cdef int j

    for i in range(maxiter):
        # get xb = matmul(x, gamma)
        dgemv('T', &p, &n, &double_one, &x[0,0], &p, &gamma[0], &one, &double_zero, &xb[0], &one)

        # get grad = matmul(tr(x), pr - y)/n
        pr = logistic(xb, n)
        err = vecsub(pr, y, n)

        dgemv('N', &p, &n, &double_one, &x[0,0], &p, &err[0], &one, &double_zero, &grad[0], &one)
        dscal(&p, &n_inv, &grad[0], &one)

        # compute L-inf Ball Projection of matmul(F, gamma)/mu
        dgemv('T', &p, &m, &double_one, &F[0,0], &p, &gamma[0], &one, &double_zero, &a[0], &one)
        dscal(&m, &mu_inv, &a[0], &one)

        for j in range(m):
            a[j] = LinfBallProjection(a[j])

        # compute matmul(tr(F), a)
        dgemv('N', &p, &m, &double_one, &F[0,0], &p, &a[0], &one, &double_zero, &grad_F[0], &one)

        # add two gradients
        grad = vecadd(grad, grad_F, p)

        # multiply by step size
        dscal(&p, &t, &grad[0], &one)

        # apply gradient
        alpha = vecsub(gamma, grad, p)

        dcopy(&p, &beta[0], &one, &_beta[0], &one)
        dcopy(&p, &alpha[0], &one, &beta[0], &one)

        for j in range(1, p):
            beta[j] = soft_thresholding(alpha[j], lam)

        if vecnorm(vecsub(_beta, beta, p), p) < tol:
            break

        s[1] = (1 + sqrt(1 + 4 * s[0] * s[0])) * 0.5
        mm = (s[0] - 1) / s[1]

        momentum = vecsub(beta, _beta, p)
        dscal(&p, &mm, &momentum[0], &one)
        gamma = vecadd(beta, momentum, p)

        s[0] = s[1]

    return np.array(beta)


@cython.boundscheck(False)
@cython.wraparound(False)
def joint_fair_logistic_solver(double[:,:] x_1, double[:,:] x_2, double[:] y_1, double[:] y_2,
                               double[:] D0, double[:] D1, double t, double mu,
                               double[:] lam1, double lam4, int maxiter, double tol):
    # Note: x_1 is an array of size (n_1, p) but its transpose x_1.T of size (p, n_1) is passed to this function (for blas subroutine)
    # Note: x_2 is an array of size (n_2, p) but its transpose x_2.T of size (p, n_2) is passed to this function (for blas subroutine)

    cdef int n_1 = x_1.shape[1]
    cdef int n_2 = x_2.shape[1]
    cdef int p = x_1.shape[0]

    cdef int pp = 2*p

    cdef double[:] beta = np.zeros(pp)
    cdef double[:] _beta = np.zeros(pp)
    cdef double[:] alpha = np.zeros(pp)
    cdef double[:] gamma = np.zeros(pp)
    cdef double[:] grad = np.zeros(pp)
    cdef double[:] grad_F = np.zeros(pp)
    cdef double[:] momentum = np.zeros(pp)

    cdef double a_0
    cdef double a_1
    cdef double[:] a_2 = np.zeros(p)
    cdef double[:] s = np.ones(2)
    cdef double[:] xb_1 = np.zeros(n_1)
    cdef double[:] xb_2 = np.zeros(n_2)
    cdef double[:] err_1 = np.zeros(n_1)
    cdef double[:] err_2 = np.zeros(n_2)
    cdef double[:] pr_1 = np.zeros(n_1)
    cdef double[:] pr_2 = np.zeros(n_2)

    cdef double double_zero = 0
    cdef double double_one = 1
    cdef double negative_one = -1
    cdef int one = 1
    cdef double mm = 1
    cdef double mu_inv = 1/mu
    cdef double n_1_inv = 1/np.float64(n_1)
    cdef double n_2_inv = 1/np.float64(n_2)

    cdef int i
    cdef int j

    for i in range(maxiter):
        # get xb = matmul(x, gamma)
        dgemv('T', &p, &n_1, &double_one, &x_1[0,0], &p, &gamma[0], &one, &double_zero, &xb_1[0], &one)
        dgemv('T', &p, &n_2, &double_one, &x_2[0,0], &p, &gamma[p], &one, &double_zero, &xb_2[0], &one)

        pr_1 = logistic(xb_1, n_1)
        pr_2 = logistic(xb_2, n_2)

        err_1 = vecsub(pr_1, y_1, n_1)
        err_2 = vecsub(pr_2, y_2, n_2)

        dgemv('N', &p, &n_1, &double_one, &x_1[0,0], &p, &err_1[0], &one, &double_zero, &grad[0], &one)
        dgemv('N', &p, &n_2, &double_one, &x_2[0,0], &p, &err_2[0], &one, &double_zero, &grad[p], &one)

        dscal(&p, &n_1_inv, &grad[0], &one)
        dscal(&p, &n_2_inv, &grad[p], &one)

        a_0 = ddot(&pp, &D0[0], &one, &gamma[0], &one)*mu_inv
        a_1 = ddot(&pp, &D1[0], &one, &gamma[0], &one)*mu_inv

        for j in range(p):
            a_2[j] = gamma[j] - gamma[j+p]

        dscal(&p, &lam4, &a_2[0], &one)
        dscal(&p, &mu_inv, &a_2[0], &one)

        # apply L-inf Ball Projection on a
        a_0 = LinfBallProjection(a_0)
        a_1 = LinfBallProjection(a_1)
        for j in range(p):
            a_2[j] = LinfBallProjection(a_2[j])

        dscal(&pp, &a_0, &D0[0], &one)
        dscal(&pp, &a_1, &D1[0], &one)
        #grad_F = vecadd(D0, D1, p)

        #grad_F[:p] = vecadd(grad_F[:p], a_2, p)
        #grad_F[p:] = vecsub(grad_F[p:], a_2, p)
        dcopy(&p, &a_2[0], &one, &grad_F[0], &one)
        dcopy(&p, &a_2[0], &one, &grad_F[p], &one)
        dscal(&p, &negative_one, &grad_F[p], &one)

        dscal(&pp, &lam4, &grad_F[0], &one)

        grad_F = vecadd(grad_F, D0, pp)
        grad_F = vecadd(grad_F, D1, pp)

        grad = vecadd(grad, grad_F, pp)

        # multiply by step size
        dscal(&pp, &t, &grad[0], &one)

        # apply gradient
        alpha = vecsub(gamma, grad, pp)

        dcopy(&pp, &beta[0], &one, &_beta[0], &one)
        dcopy(&pp, &alpha[0], &one, &beta[0], &one)

        for j in range(1, p):
            beta[j] = soft_thresholding(alpha[j], lam1[0])
            beta[j+p] = soft_thresholding(alpha[j+p], lam1[1])

        if vecnorm(vecsub(_beta, beta, p), p) < tol:
            break

        s[1] = (1 + sqrt(1 + 4 * s[0] * s[0])) * 0.5
        mm = (s[0] - 1) / s[1]

        momentum = vecsub(beta, _beta, pp)
        dscal(&pp, &mm, &momentum[0], &one)
        gamma = vecadd(beta, momentum, pp)

        s[0] = s[1]

    return np.array(beta)