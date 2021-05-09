# joint-fairness-model
Implementation of the paper Joint Fairness Model (JFM) with Applications to Risk Predictions for Under-represented Populations

## Dependencies
- [anaconda3](https://www.anaconda.com/download/) (>= 4.8.3)
- [cython](https://cython.org/) (>= 0.29.8)
- [scipy](https://www.scipy.org/) (>= 1.6.2)
- [numpy](https://numpy.org/) (>= 1.17.0)
- [pandas](https://pandas.pydata.org/) (>= 1.2.4)
- [matplotlib](https://matplotlib.org/) (>= 3.1.1)
- [scikit-learn](https://scikit-learn.org/) (>= 0.24.1)

---

## Install
Users have to compile the enclosed cython source code (tested on Windows 10, macOS Catalina 10.15.7, and Red Hat Enterprise Linux 8.2.)
```
git clone https://github.com/hyungrok-do/joint-fairness-model
cd joint-fairness-model
python setup.py build_ext --inplace
```

---

## Models
We provide three models based on logistic regression. The model classes are similar to scikit-learn estimators inheriting from scikit-learn's ```BaseEstimator```.
- ``` models.LogisticLasso ``` for L1 penalized logistic regression (also known as logistic Lasso.)
- ``` models.LogisticSingleFair ``` for the single fairness model (SFM): a logistic regression penalized by L1 penalty and fairness penalty.
- ``` models.LogisticJointFair ``` for the joint fairness model (JFM): the proposed method (see paper.)

Note that ```LogisticLasso``` uses fast iterative shrinkage and thresholding algorithm (FISTA) [[1]](#1) and solvers for ```LogisticSingleFair``` and ```LogisticJointFair``` are implemented with smoothing proximal gradient method [[2]](#2).

---

## Reproducing the Simulation Results
### Main Simulation Scenarios
Running ```experiments-simulation.py``` will produce a single simulation results.

```
python experiments-simulation.py \
       --save_path path/to/save (default is the current path) \
       --name name-of-simulation (default is 'untitled') \
       --seed 55 \
       --p 100 \
       --q 40 \
       --r 20 \
       --n1 500 \
       --n2 200 \
       --b -10 \
       --t 0
```

We provide shell/slurm scripts to run the repeated experiments to reproduce the results for scenarios 1 through 4: ```run-simulation.sh``` and ```run-simulation.s```. To draw the plots, run ```visualization-simulation-results.py```.

### Supplement Simulation Scenarios
```run-simulation.sh``` and ```run-simulation.s``` will also provide the additional simulation scenarios' (1B through 4B) results.

### Validation Measures
Run ```run-validation-measures.sh``` or ```run-validation-measures.s``` to get the experimental results for the validation measures. To draw the plots, run ```visualization-validation-measures.py```.

### Computation Times
Run ```experiment-computation-time-p.py``` and ```experiment-computation-time-n.py```. Both scripts do not require any arguments.

---

## Further Work/Comments
- SFM and JFM for more than two groups (current codes only work for two groups just for reproducing the simulation results: the generalized version will be updated shortly.)
- Models without intercept term will be implemented.
- ```GridSearchCV``` of scikit-learn may not work on Windows machine (seems like multiprocessing issue.)

---

## References

<a id="1">[1]</a> Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. <em>SIAM journal on imaging sciences,</em> 2(1), 183-202.

<a id="2">[2]</a> Chen, X., Lin, Q., Kim, S., Carbonell, J. G., & Xing, E. P. (2012). Smoothing proximal gradient method for general structured sparse regression. <em>The Annals of Applied Statistics,</em> 719-752.
