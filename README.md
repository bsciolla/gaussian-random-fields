# gaussian-random-fields
Generator of 2D gaussian random fields
======================================

Generates 2D gaussian random maps.

The probability distribution of each variable follows a Normal distribution.
The variables in the map are spatially correlated.
The correlations are due to a scale-free spectrum P(k) ~ 1/|k|^(alpha/2).

The library uses Numpy+Scipy.
See the Notebook demo:
- [demo.ipynb](demo.ipynb)

Or run with:
```
 python gaussian_random_fields.py
```
