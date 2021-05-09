
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from .solvers import logistic_lasso_gradient_solver
from .solvers import single_fair_logistic_solver
from .solvers import joint_fair_logistic_solver

global MAXITER
MAXITER = 1000

class LogisticLasso(BaseEstimator):
    """ Logistic regression model with L1 penalty

    Parameters
    ----------
    lam : float, default=0.1
        Hyperparameter that multiplies the L1 penalty term. Defaults to 0.1.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for logistic lasso model.

    maxiter : int, default=1000
        The maximum number of iterations for the numerical optimization algorithm.

    tol : float, default=1e-3
        The tolerance for the numerical optimization algorithm. Terminates the algorithm if the updates are smaller than ``tol``.

    Attributes
    ----------
    coef_ : numpy.ndarray of shape (n_features,)
        Estimated coefficients for logistic lasso model.
    """
    def __init__(self, lam=0.1, fit_intercept=True, maxiter=MAXITER, tol=1e-3):
        assert lam > 0, 'parameter lam must be positive real value'

        self.lam = lam
        self.maxiter = maxiter
        self.fit_intercept = True
        self.tol = tol
        self.coef_ = None

    def fit(self, X, y):
        """ Fit LogisticLasso model with FISTA (Beck & Teboulle, 2009)

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Covariate matrix.

        y : numpy.ndarray of shape (n_samples,)
            Target vector.

        """
        if type(y) == pd.DataFrame:
            y = y.values.flatten()
        y = y.astype(np.float64)

        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])

        n = len(y)
        t = 4 / np.max(np.linalg.eigvalsh(X.T.dot(X)))

        lam = np.double(self.lam*t)
        maxiter = np.int(self.maxiter)
        tol = np.double(self.tol)
        t = np.double(t / n)

        beta = logistic_lasso_gradient_solver(X.T, y, t, lam, maxiter, tol)

        self.coef_ = beta

    def predict_proba(self, X, y=None):
        """ Predict probability estimates.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Covariate matrix.

        y : Not used.

        Returns
        ----------
        P : numpy.ndarray of shape (n_samples, 2)
            Estimated probability matrix.

        """
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        xb = X.dot(self.coef_)
        p = 1. / (1. + np.exp(-xb))
        return np.column_stack([1. - p, p])

    def predict(self, X, y=None):
        """ Predict probability estimates.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Covariate matrix.

        y : Not used.

        Returns
        ----------
        P : numpy.ndarray of shape (n_samples,)
            Predicted class vector
        """
        p = self.predict_proba(X, y)
        return (p[:,1] > 0.5).astype(np.int)


class LogisticSingleFair(BaseEstimator):
    """ Logistic Single Fairness model with L1 and Fairness Penalty.

    Parameters
    ----------
    lam1 : float, default=0.1
        Hyperparameter that multiplies the L1 penalty term. Defaults to 0.1.

    lam2 : float, default=0.1
        Hyperparameter that multiplies the linearized equalized odds penalty terms. Defaults to 0.1.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for logistic lasso model.

    maxiter : int, default=1000
        The maximum number of iterations for the numerical optimization algorithm.

    mu : float, default=1e-3
        The approximation parameter for the numerical optimization algorithm. Smaller values provide better approximation,
        but too small values may lead to numerical instability.

    tol : float, default=1e-3
        The tolerance for the numerical optimization algorithm. Terminates the algorithm if the updates are smaller than ``tol``.

    Attributes
    ----------
    coef_ : numpy.ndarray of shape (n_features,)
        Estimated coefficients for logistic lasso model.
    """

    def __init__(self, lam1=0.1, lam2=0.1, protected_idx=0, fit_intercept=True, maxiter=MAXITER, mu=1e-3, tol=1e-3):
        assert lam1 > 0, 'parameter lam1 must be positive real value'
        assert lam2 > 0, 'parameter lam2 must be positive real value'

        self.lam1 = lam1
        self.lam2 = lam2
        self.protected_idx = protected_idx
        self.fit_intercept = True
        self.maxiter = maxiter
        self.mu = mu
        self.tol = tol

    def fit(self, X, y):
        """ Fit LogisticSingleFair model with Accelerated Smoothing Proximal Gradient (Chen et al., 2012)

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Covariate matrix.

        y : numpy.ndarray of shape (n_samples,)
            Target vector.

        """
        if type(y) == pd.DataFrame:
            y = y.values.flatten()
        A = X[:, self.protected_idx]
        X = np.delete(X, self.protected_idx, axis=1)
        X = np.column_stack([np.ones(len(X)), X])
        y = y.astype(np.float64)
        n = len(y)

        x_0, x_1, y_0, y_1 = self._split_by_protected_value(A, X, y)

        y_1 = y_1.astype(np.float64)
        y_0 = y_0.astype(np.float64)

        F = np.row_stack([
            self.lam2 * (x_0[y_0 == 0].mean(0) - x_1[y_1 == 0].mean(0)),
            self.lam2 * (x_0[y_0 == 1].mean(0) - x_1[y_1 == 1].mean(0))])

        _a, sx, _b = np.linalg.svd(F, False)
        L = np.max(np.linalg.eigvalsh(np.dot(X.T, X) / n))/4 + (np.max(sx) ** 2) / self.mu
        t = 1 / L

        lam1 = np.double(self.lam1*t)
        maxiter = np.int(self.maxiter)
        tol = np.double(self.tol)
        t = np.double(t)
        mu = np.double(self.mu)

        beta = single_fair_logistic_solver(X.T, y, F.T, t, mu, lam1, maxiter, tol)

        self.coef_ = beta

    def predict_proba(self, X, y=None):
        """ Predict probability estimates.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Covariate matrix.

        y : Not used.

        Returns
        ----------
        P : numpy.ndarray of shape (n_samples, 2)
            Estimated probability matrix.

        """
        X = np.delete(X, self.protected_idx, axis=1)
        X = np.column_stack([np.ones(len(X)), X])
        xb = X.dot(self.coef_)
        p = 1. / (1. + np.exp(-xb))
        return np.column_stack([1. - p, p])

    def predict(self, X, y=None):
        """ Predict probability estimates.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Covariate matrix.

        y : Not used.

        Returns
        ----------
        P : numpy.ndarray of shape (n_samples,)
            Predicted class vector
        """
        p = self.predict_proba(X, y)
        return np.argmax(p, axis=1).astype(np.int)

    @staticmethod
    def _split_by_protected_value(A, X, y=None):
        x_1 = X[A == 1]
        x_0 = X[A == 0]

        if y is None:
            return x_0, x_1

        y_1 = y[A == 1]
        y_0 = y[A == 0]

        return x_0, x_1, y_0, y_1


class LogisticJointFair(BaseEstimator):
    def __init__(self, lam1=0.1, lam2=0.1, lam3=0.1, lam4=0.1, protected_idx=0, fit_intercept=True, maxiter=MAXITER, tol=1e-3, mu=1e-3):
        """ Logistic Joint Fairness model with L1 and Fairness Penalty.

        Currently, this class only provides two-group cases to reproduce the simulation results presented in the paper.

        Parameters
        ----------
        lam1 : float, default=0.1
            Hyperparameter that multiplies the L1 penalty term for group 1. Defaults to 0.1.

        lam2 : float, default=0.1
            Hyperparameter that multiplies the L1 penalty term for group 2. Defaults to 0.1.

        lam3 : float, default=0.1
            Hyperparameter that multiplies the linearized equalized odds penalty terms. Defaults to 0.1.

        lam4 : float, default=0.1
            Hyperparameter that multiplies the similarity penalty that encourages similar estimated coefficients
            across the groups. Defaults to 0.1.

        fit_intercept : bool, default=True
            Whether to calculate the intercept for logistic lasso model.

        maxiter : int, default=1000
            The maximum number of iterations for the numerical optimization algorithm.

        mu : float, default=1e-3
            The approximation parameter for the numerical optimization algorithm. Smaller values provide better approximation,
            but too small values may lead to numerical instability.

        tol : float, default=1e-3
            The tolerance for the numerical optimization algorithm. Terminates the algorithm if the updates are smaller than ``tol``.

        Attributes
        ----------
        coef_ : numpy.ndarray of shape (n_features,)
            Estimated coefficients for logistic lasso model.
        """

        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.lam4 = lam4
        self.protected_idx = protected_idx
        self.fit_intercept = True
        self.maxiter = maxiter
        self.tol = tol
        self.mu = mu

    def fit(self, X, y):
        if type(y) == pd.DataFrame:
            y = y.values.flatten()
        A = X[:, self.protected_idx]
        X = np.delete(X, self.protected_idx, axis=1)
        X = np.column_stack([np.ones(len(X)), X])
        y = y.astype(np.float64)

        p = X.shape[1]

        x_1, x_2, y_1, y_2 = self._split_by_protected_value(A, X, y)
        F = self.lam4 * np.column_stack([np.identity(p), -np.identity(p)])

        n_1, n_2 = float(len(y_1)), float(len(y_2))

        D0 = self.lam3 * np.concatenate([x_1[y_1 == 0].mean(0), -x_2[y_2 == 0].mean(0)])
        D1 = self.lam3 * np.concatenate([x_1[y_1 == 1].mean(0), -x_2[y_2 == 1].mean(0)])
        D = np.row_stack([D0, D1, F])
        L1 = np.max([
            np.max(np.linalg.eigvalsh(np.dot(x_1.T, x_1) / n_1)),
            np.max(np.linalg.eigvalsh(np.dot(x_2.T, x_2) / n_2))]) / 4
        L2 = np.max(np.linalg.eigvalsh(D.T.dot(D))) / self.mu
        t = 1 / (L1 + L2)
        del D, F

        lam1 = np.array([self.lam1, self.lam2])*t
        lam4 = np.double(self.lam4)
        maxiter = np.int(self.maxiter)
        tol = np.double(self.tol)
        mu = np.double(self.mu)
        t = np.double(t)
        beta = joint_fair_logistic_solver(x_1.T, x_2.T, y_1, y_2, D0, D1, t, mu, lam1, lam4, maxiter, tol)
        beta = beta.reshape(2, -1)

        self.coef_ = beta
        self.coef_0_ = beta[0]
        self.coef_1_ = beta[1]

    def predict_proba(self, X, y=None):
        """ Predict probability estimates.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Covariate matrix.

        y : Not used.

        Returns
        ----------
        P : numpy.ndarray of shape (n_samples, 2)
            Estimated probability matrix.

        """
        A = X[:, self.protected_idx]
        X = np.delete(X, self.protected_idx, axis=1)
        X = np.column_stack([np.ones(len(X)), X])
        p = np.array([1. / (1. + np.exp(-_x.dot(self.coef_[int(_a)]))) for _x, _a in zip(X, A)])
        return np.column_stack([1. - p, p])

    def predict(self, X, y=None):
        """ Predict probability estimates.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Covariate matrix.

        y : Not used.

        Returns
        ----------
        P : numpy.ndarray of shape (n_samples,)
            Predicted class vector
        """
        p = self.predict_proba(X, y)
        return np.argmax(p, axis=1).astype(np.int)

    @staticmethod
    def _split_by_protected_value(A, X, y=None):
        x_1 = X[A == 1]
        x_0 = X[A == 0]

        if y is None:
            return x_0, x_1

        y_1 = y[A == 1]
        y_0 = y[A == 0]

        return x_0, x_1, y_0, y_1
