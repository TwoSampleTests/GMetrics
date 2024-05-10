import os
import sys
import sklearn # type: ignore
import numpy as np
import pandas as pd # type: ignore
from matplotlib import pyplot as plt # type: ignore
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
import random
from typing import List, Tuple, Dict, Callable, Union, Optional
sys.path.insert(0,'..')
import GMetrics

def MixMultiNormal(ncomp: int = 3,
                   ndims: int = 4,
                   loc_factor = 1.,
                   scale_factor = 1.,
                   dtype = tf.float64,
                   seed: int = 0
                  ) -> tfp.distributions.Mixture:
    GMetrics.utils.reset_random_seeds(seed)
    loc: tf.Tensor = tf.random.uniform([ncomp, ndims], 
                                        minval = -loc_factor, 
                                        maxval = loc_factor, 
                                        dtype = dtype)
    scale: tf.Tensor = tf.random.uniform([ncomp, ndims], 
                                          minval = 0, 
                                          maxval = scale_factor, 
                                          dtype = dtype)
    probs: tf.Tensor = tf.random.uniform([ncomp], 
                                          minval = 0, 
                                          maxval = 1, 
                                          dtype = dtype)
    probs = probs / tf.reduce_sum(probs)
    components = tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale)
    mix_gauss = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(probs=probs),
        components_distribution=components,
        validate_args=True)
    return mix_gauss

def MultiNormalFromMix(ncomp: int = 3,
                       ndims: int = 4,
                       loc_factor = 1.,
                       scale_factor = 1.,
                       dtype = tf.float64,
                       seed: int = 0
                      ) -> tfp.distributions.MultivariateNormalTriL: 
    GMetrics.utils.reset_random_seeds(seed)
    loc: tf.Tensor = tf.random.uniform([ndims], 
                                       minval = -loc_factor,
                                       maxval = loc_factor, 
                                       dtype = dtype)
    mix = MixMultiNormal(ncomp = ncomp,
                         ndims = ndims,
                         loc_factor = loc_factor,
                         scale_factor = scale_factor,
                         dtype = dtype,
                         seed = seed)
    covariance_matrix = mix.covarianc()
    scale: tf.Tensor = tf.linalg.cholesky(covariance_matrix) # type: ignore
    mvn = tfp.distributions.MultivariateNormalTriL(loc = loc, scale_tril = scale)
    return mvn

def deform_mean(distribution,
                eps = 0.,
                seed: int = 0):
    if eps < 0:
        raise ValueError("Epsilon must be non-negative")
    if float(eps) == 0.:
        return distribution
    else:
        GMetrics.utils.reset_random_seeds(seed)
        shape = distribution.mean().shape
        dtype = distribution.mean().dtype
        shift_vector = tf.random.uniform(shape, 
                                         minval = -eps, 
                                         maxval = eps, 
                                         dtype = dtype)
        deformed_dist = tfd.TransformedDistribution(distribution = distribution, 
                                                    bijector = tfb.Shift(shift = shift_vector))
        return deformed_dist
        
def deform_cov_diag(distribution, 
                    eps=0.,
                    seed: int = 0):
    if eps < 0.:
        raise ValueError("Epsilon must be non-negative")
    if float(eps) == 0.:
        return distribution
    else:
        GMetrics.utils.reset_random_seeds(seed)
        shape = distribution.mean().shape
        dtype = distribution.mean().dtype
        scale_vector = tf.random.uniform(shape, minval=1., maxval=1. + eps, dtype=dtype)
        original_mean = distribution.mean()
        shift_to_zero = tfb.Shift(-original_mean)  # Shift distribution so mean is at zero
        scale = tfb.Scale(scale_vector)            # Scale the distribution
        shift_back = tfb.Shift(original_mean)      # Shift distribution back to original mean
        chained_bijector = tfb.Chain([shift_back, scale, shift_to_zero])
        deformed_dist = tfd.TransformedDistribution(distribution=distribution, 
                                                    bijector=chained_bijector)
        return deformed_dist

def modify_covariance_matrix(original_covariance, eps):
    if eps < 0:
        raise ValueError("Epsilon must be non-negative")
    dtype = original_covariance.dtype
    shape = original_covariance.shape[0]
    std_devs = tf.sqrt(tf.linalg.diag_part(original_covariance))
    modified_std_devs_diag = std_devs / tf.maximum(tf.constant(1.0, dtype=dtype), tf.constant(eps, dtype=dtype))
    modified_std_devs_off_diag = std_devs * tf.maximum(tf.constant(0.0, dtype=dtype), tf.constant(1 - eps, dtype=dtype))
    correlation_matrix = original_covariance / (std_devs[:, None] * std_devs[None, :])
    modified_diag = tf.linalg.diag(modified_std_devs_diag**2)
    outer_std_devs = modified_std_devs_off_diag[:, None] * modified_std_devs_off_diag[None, :]
    modified_off_diag = correlation_matrix * outer_std_devs
    diagonal_mask = tf.linalg.diag(tf.ones(shape, dtype=dtype))
    modified_off_diag = modified_off_diag * (1 - diagonal_mask)
    modified_covariance = modified_diag + modified_off_diag
    return modified_covariance

def deform_cov_off_diag(distribution,
                        eps = 0.,
                        seed = 0.,
                        nsamples: int = 100_000):
    if eps < 0.:
        raise ValueError("Epsilon must be non-negative")
    if float(eps) == 0.:
        return distribution
    else:
        GMetrics.utils.reset_random_seeds(seed)
        dtype = distribution.mean().dtype
        try:
            original_covariance = distribution.covariance()
        except:
            samp = tf.cast(distribution.sample(nsamples), dtype = dtype)
            original_covariance = tfp.stats.covariance(samp, sample_axis = 0)
        modified_covariance = modified_covariance = modify_covariance_matrix(original_covariance, eps)
        chol_original = tf.linalg.cholesky(original_covariance)
        chol_modified = tf.linalg.cholesky(modified_covariance)
        transformation_matrix_transpose = tf.linalg.triangular_solve(tf.linalg.matrix_transpose(chol_original), 
                                                                     tf.linalg.matrix_transpose(chol_modified),
                                                                     lower=False)
        transformation_matrix = tf.linalg.matrix_transpose(transformation_matrix_transpose)
        original_mean = distribution.mean()
        shift_to_zero = tfb.Shift(-original_mean)
        linear_transform = tfb.ScaleMatvecTriL(scale_tril=transformation_matrix)
        shift_back = tfb.Shift(original_mean)
        chained_bijector = tfb.Chain([shift_back, linear_transform, shift_to_zero])
        deformed_dist = tfd.TransformedDistribution(distribution=distribution, bijector=chained_bijector)
        return deformed_dist

class AbsPowerTransform(tfb.Bijector):
    def __init__(self, power = 1., validate_args=False, name="sign_safe_power_transform"):
        super(AbsPowerTransform, self).__init__(forward_min_event_ndims=0, validate_args=validate_args, name=name)
        self.power = power

    def _forward(self, x):
        # Apply power transformation only to the absolute value and keep the sign
        power = tf.cast(self.power, x.dtype)
        return tf.sign(x) * tf.pow(tf.abs(x), power)

    def _inverse(self, y):
        # Inverse transformation, assuming y has the same sign as x
        power = tf.cast(self.power, y.dtype)
        return tf.sign(y) * tf.pow(tf.abs(y), tf.cast(1., y.dtype) / power) # type: ignore

    def _forward_log_det_jacobian(self, x):
        # Logarithm of the absolute value of the derivative of the forward transformation
        power = tf.cast(self.power, x.dtype)
        return (power - tf.cast(1.,x.dtype)) * tf.math.log(tf.abs(x)) + tf.math.log(tf.abs(power)) # type: ignore
    
def deform_power_abs(distribution,
                     eps = 0.,
                     direction = "up"):
    if eps < 0.:
        raise ValueError("Epsilon must be non-negative")
    if float(eps) == 0.:
        return distribution
    else:
        if direction == "up":
            deformation = AbsPowerTransform(power = 1 + eps)
        elif direction == "down":
            deformation = AbsPowerTransform(power = 1 - eps)
        else:
             raise ValueError("Direction must be either 'up' or 'down'")
        deformed_dist = tfd.TransformedDistribution(distribution = distribution, 
                                                    bijector = deformation)
        return deformed_dist

class RandomShift(tfb.Bijector):
    def __init__(self, scale=0.1, shift_dist="normal", validate_args=False, name="random_shift"):
        super(RandomShift, self).__init__(forward_min_event_ndims=0, validate_args=validate_args, name=name)
        self.scale = scale
        if shift_dist in ["normal", "uniform"]:
            self.shift_dist = shift_dist
        else:
            raise ValueError("shift_dist must be either 'normal' or 'uniform'")

    def _forward(self, x):
        # Ensuring that the random shift has the same dtype as the input x
        if self.shift_dist == "normal":
            shift = tf.random.normal(shape=tf.shape(x), stddev=self.scale, dtype=x.dtype)
        elif self.shift_dist == "uniform":
            shift = tf.random.uniform(shape=tf.shape(x), minval=-self.scale, maxval=self.scale, dtype=x.dtype)
        return x + shift

    def _inverse(self, y):
        # This is not truly correct as we don't know the original shift
        raise NotImplementedError("Inverse is not well defined for random shifts.")

    def _forward_log_det_jacobian(self, x):
        # The log determinant of the Jacobian for a pure shift is zero
        return tf.zeros_like(x)

def deform_random(distribution,
                  eps = 0.,
                  shift_dist = "normal",
                  seed: int = 0):
    if eps < 0:
        raise ValueError("Epsilon must be non-negative")
    if float(eps) == 0.:
        return distribution
    else:
        GMetrics.utils.reset_random_seeds(seed)
        deformation = RandomShift(scale = eps,
                                  shift_dist = shift_dist)
        deformed_dist = tfd.TransformedDistribution(distribution = distribution, 
                                                    bijector = deformation)
        return deformed_dist

def deformed_distribution(distribution,
                          eps = 0.,
                          deform_type = "mean",
                          **deform_kwargs):
    """
    Valid deformations are 'mean', 'cov_diag', 'cov_off_diag', 'power_abs', 'random'
    default kwargs for each deformation are
    
        - 'mean': {seed: 0}
        - 'cov_diag': {seed: 0}
        - 'cov_off_diag': {seed: 0, nsamples: 100_000}
        - 'power_abs': {direction: 'up'} # alternative value direction: 'down'
        - 'random': {shift_dist: 'normal', seed: 0} # alternative value shift_dist: 'uniform'
        
    """
    if deform_type not in ["mean", "cov_diag", "cov_off_diag", "power_abs", "random"]:
        raise ValueError("Deformation type must be one of 'mean', 'cov_diag', 'cov_off_diag', 'power_abs', 'random'")
    func: Callable = eval("deform_"+deform_type)
    deformed_dist = func(distribution, eps = eps, **deform_kwargs)
    return deformed_dist

def describe_distributions(distributions: List[tfp.distributions.Distribution]) -> None:
    """
    Describes a 'tfp.distributions' object.
    
    Args:
        distributions: list of 'tfp.distributions' objects, distributions to describe

    Returns:
        None (prints the description)
    """
    print('\n'.join([str(d) for d in distributions]))

def rot_matrix(data: np.ndarray) -> np.ndarray:
    """
    Calculates the matrix that rotates the covariance matrix of 'data' to the diagonal basis.

    Args:
        data: np.ndarray, data to rotate

    Returns:
        rotation: np.ndarray, rotation matrix
    """
    cov_matrix: np.ndarray = np.cov(data, rowvar=False)
    w: np.ndarray
    V: np.ndarray
    w, V = np.linalg.eig(cov_matrix)
    return V

def transform_data(data: np.ndarray,
                   rotation: np.ndarray) -> np.ndarray:
    """
    Transforms the data according to the rotation matrix 'rotation'.
    
    Args:
        data: np.ndarray, data to transform
        rotation: np.ndarray, rotation matrix

    Returns:
        data_new: np.ndarray, transformed data
    """
    if len(rotation.shape) != 2:
        raise ValueError('Rottion matrix must be a 2D matrix.')
    elif rotation.shape[0] != rotation.shape[1]:
        raise ValueError('Rotation matrix must be square.')
    data_new: np.ndarray = np.dot(data,rotation)
    return data_new

def inverse_transform_data(data: np.ndarray,
                           rotation: np.ndarray) -> np.ndarray:
    """
    Transforms the data according to the inverse of the rotation matrix 'rotation'.
    
    Args:
        data: np.ndarray, data to transform
        rotation: np.ndarray, rotation matrix
        
    Returns:
        data_new: np.ndarray, transformed data
    """
    if len(rotation.shape) != 2:
        raise ValueError('Rottion matrix must be a 2D matrix.')
    elif rotation.shape[0] != rotation.shape[1]:
        raise ValueError('Rotation matrix must be square.')
    data_new: np.ndarray = np.dot(data,np.transpose(rotation))
    return data_new

def reset_random_seeds(seed: int = 0) -> None:
    """
    Resets the random seeds of the packages 'tensorflow', 'numpy' and 'random'.
    
    Args:
        seed: int, random seed
        
    Returns:
        None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def RandCorr(matrixSize: int,
             seed: int) -> np.ndarray:
    """
    Generates a random correlation matrix of size 'matrixSize' x 'matrixSize'.

    Args:
        matrixSize: int, size of the matrix
        seed: int, random seed
        
    Returns:
        Vnorm: np.ndarray, normalized random correlation matrix
    """
    np.random.seed(0)
    V: np.ndarray = sklearn.datasets.make_spd_matrix(matrixSize,
                                                     random_state = seed)
    D: np.ndarray = np.sqrt(np.diag(np.diag(V)))
    Dinv: np.ndarray = np.linalg.inv(D)
    Vnorm: np.ndarray = np.matmul(np.matmul(Dinv,V),Dinv)
    return Vnorm

def is_pos_def(x: np.ndarray) -> bool:
    """ 
    Checks if the matrix 'x' is positive definite.
    
    Args:
        x: np.ndarray, matrix to check

    Returns:
        bool, True if 'x' is positive definite, False otherwise
    """
    if len(x.shape) != 2:
        raise ValueError('Input to is_pos_def must be a 2-dimensional array.')
    elif x.shape[0] != x.shape[1]:
        raise ValueError('Input to is_pos_def must be a square matrix.')
    return bool(np.all(np.linalg.eigvals(x) > 0))

def RandCov(std: np.ndarray,
            seed: int) -> np.ndarray:
    """
    Generates a random covariance matrix of size 'matrixSize' x 'matrixSize'.

    Args:
        std: np.ndarray, standard deviations of the random variables
        seed: int, random seed
        
    Returns:
        V: np.ndarray, random covariance matrix
    """
    matrixSize: int = len(std)
    corr: np.ndarray = RandCorr(matrixSize,seed)
    D: np.ndarray = np.diag(std)
    V: np.ndarray = np.matmul(np.matmul(D,corr),D)
    return V