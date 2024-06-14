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
    covariance_matrix = mix.covariance()
    scale: tf.Tensor = tf.linalg.cholesky(covariance_matrix) # type: ignore
    mvn = tfp.distributions.MultivariateNormalTriL(loc = loc, scale_tril = scale)
    return mvn

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