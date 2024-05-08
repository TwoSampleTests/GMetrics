# GMetrics
Module with Numpy/Scipy and TensorFlow2 implementation of several metrics for
two-sample-tests and Generative models evaluation. The implementation allows to parallelize
the calculation of null-hypotheses, speeding up inference.

The module supports the following non-parammetric two sample test statistics:

    - Dimension averaged Kolmogorov-Smirnov (KS) test-statistic and p-value
    - Sliced KS test-statistic
    - Sliced Wasserstein Distance (SWD) test-statistic
    - Frobenius norm of the difference between the correlation matrices (FN) test-statistic
    - Frechét Gaussian Distance (FGD)
    - Maximum Mean Discrepacy (MMD) (the unbiased squared estimator with a polynomial 
    kernel of arbitrary degree is implemented)

The module also includes the parametric test-statistic for the log-likelihood ratio (LLR) 
(computed from the known PDFs), that, when both PDFs are known, 
can be used as proxy of "best estimator" by the Neyman-Pearson lemma.

The module includes the following files and folders:

    - __init__.py 
        Package initialization

    - base.py
        Base classes for test object, input data parsing, and results

    - metrics/fgd.py
        Implementation of the FGD metric
    
    - metrics/fn.py
        Implementation of the FN metric

    - metrics/ks.py
        Implementation of the dimension averaged KS metric

    - metrics/llr.py
        Implementation of the LLR metric

    - metrics/mmd.py
        Implementation of the MMD metric
    
    - metrics/sks.py
        Implementation of the SKS metric

    - metrics/swd.py
        Implementation of the SWD metric

    - ultils.py
        File including utility functions

    - notebooks/
        Folder including Jupyter notebooks  with sample code

    - more/
        Folder with additional utilities for generating mixture distributions and deformations,
        and for parameters inference
