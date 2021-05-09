# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:00:00 2021

@author: Hyungrok Do
         hyungrok.do11@gmail.com
         https://github.com/hyungrok-do/

Implementation of the paper Joint Fairness Model with Applications to Risk Predictions for Under-represented Populations

"""

import numpy as np

class DataGenerator(object):
    """ Data generator for two-group unfairness data simulation.

    Parameters
    ----------
    p : int
        The number of covariates.

    q : int
        The number of non-zero coefficients in the linear model. The first ```q``` consecutive coefficients will be
        non-zero for the first group and ```q``` consecutive coefficients from ```r``` to ```r+q-t``` will be non-zero
        for the second group.

    r : int
        The position of the first non-zero coefficient for the second group. The coefficients located between ```r```
        and ```r+q-t``` will have non-zero values for the second group. Thus, ```r=0``` defines identical true models
        for two groups. On the other hand, ```r=q``` defines completely separated set of non-zero coefficients.

    b : float
        The intercept term for the true model of Group 2 controls the baseline prevalence of Group 2. A greater value
        of ```b``` means the higher baseline prevalence for Group 2.

    t : int, default=0
        The position of the last non-zero coefficient for the second group.  The coefficients located between ```r```
        and ```r+q-t``` will have non-zero values for the second group.

    Attributes
    ----------
    b1 : numpy.ndarray of shape (p,)
        True coefficients for Group 1.

    b2 : numpy.ndarray of shape (p,)
        True coefficients for Group 2.
    """

    def __init__(self, p, q, r, b, t=0):
        self._p = p
        self._q = q
        self._r = r
        self._b = b
        self._t = t

        b1 = np.zeros(p)
        b2 = np.zeros(p)

        b1[:q] = 3
        b2[r:r + q - t] = 3

        b1 = np.concatenate([[0], b1])
        b2 = np.concatenate([[b], b2])

        self.b1 = b1
        self.b2 = b2

    def generate(self, n1, n2, seed):
        """ Generate data based on the given scenario settings.

        All the covariates are drawn from i.i.d standard normal distribution. The first column is a group indicator.

        Parameters
        ----------
        n1 : int
            The number of samples for the first group.

        n2 : int
            The number of samples for the second group.

        seed : int
            Random seed.

        Returns
        ----------
        X : numpy.ndarray of shape (n1+n2, p+1)
            The covariate matrix. The first column is a group indicator: 0 for group 1 and 1 for group 2.

        y : numpy.ndarray of shape (n1+n2,)
            The response variable vector.

        """
        np.random.seed(seed)

        x1 = np.random.normal(0, 1, (n1, self._p))
        x1 = np.column_stack([np.zeros(n1), x1])
        p1 = 1. / (1 + np.exp(-x1.dot(self.b1)))

        x2 = np.random.normal(0, 1, (n2, self._p))
        x2 = np.column_stack([np.ones(n2), x2])
        p2 = 1. / (1 + np.exp(-x2.dot(self.b2)))

        y1 = np.random.binomial(1, p1)
        y2 = np.random.binomial(1, p2)

        x = np.row_stack([x1, x2])
        y = np.concatenate([y1, y2])

        idx = np.arange(len(x))
        np.random.shuffle(idx)
        x = x[idx]
        y = y[idx]

        return x, y
