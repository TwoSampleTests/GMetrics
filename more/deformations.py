import sys
import numpy as np
import pandas as pd # type: ignore
from matplotlib import pyplot as plt # type: ignore
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from typing import Callable, Union
sys.path.insert(0,'..')
import GMetrics
    
def deform_mean(d: Union[tfp.distributions.Distribution, tf.Tensor],
                eps: float = 0.,
                seed: int = 0
               ) -> Union[tfp.distributions.Distribution, tf.Tensor]:
    if eps < 0:
        raise ValueError("Epsilon must be non-negative")
    if float(eps) == 0.:
        return d
    GMetrics.utils.reset_random_seeds(seed)
    if isinstance(d, tf.Tensor):
        shape = tf.reduce_mean(d, axis=0).shape
        dtype = tf.reduce_mean(d, axis=0).dtype
    elif isinstance(d, tfd.Distribution):
        shape = d.mean().shape
        dtype = d.mean().dtype
    else:
        raise ValueError("Input must be either a tf.Tensor or a tfd.Distribution")
    shift_vector = tf.random.uniform(shape,
                                     minval = -1 * eps,
                                     maxval = eps,
                                     dtype = dtype)
    if isinstance(d, tf.Tensor):
        deformed_d = d + shift_vector
    elif isinstance(d, tfd.Distribution):
        deformed_d = tfd.TransformedDistribution(distribution = d,
                                                 bijector = tfb.Shift(shift = shift_vector))
    return deformed_d
    
def deform_cov_diag(d: Union[tfp.distributions.Distribution, tf.Tensor],
                    eps: float = 0.,
                    seed: int = 0
                   ) -> Union[tfp.distributions.Distribution, tf.Tensor]:
    if eps < 0:
        raise ValueError("Epsilon must be non-negative")
    if float(eps) == 0.:
        return d
    GMetrics.utils.reset_random_seeds(seed)
    if isinstance(d, tf.Tensor):
        shape = tf.reduce_mean(d,axis=0).shape
        dtype = tf.reduce_mean(d,axis=0).dtype
        original_mean = tf.reduce_mean(d,axis=0)
    elif isinstance(d, tfd.Distribution):
        shape = d.mean().shape
        dtype = d.mean().dtype
        original_mean = d.mean()
    else:
        raise ValueError("Input must be either a tf.Tensor or a tfd.Distribution")
    scale_vector = tf.random.uniform(shape, 
                                     minval = 1., 
                                     maxval = 1. + eps, 
                                     dtype = dtype)
    shift_to_zero = tfb.Shift(-original_mean)
    scale = tfb.Scale(scale_vector)
    shift_back = tfb.Shift(original_mean)
    chained_bijector = tfb.Chain([shift_back, scale, shift_to_zero])
    if isinstance(d, tf.Tensor):
        deformed_d = chained_bijector.forward(d)
    elif isinstance(d, tfd.Distribution):
        deformed_d = tfd.TransformedDistribution(distribution = d,
                                                 bijector = chained_bijector)
    return deformed_d

#def modify_covariance_matrix(original_covariance, eps):
#    if eps < 0:
#        raise ValueError("Epsilon must be non-negative")
#    dtype = original_covariance.dtype
#    shape = original_covariance.shape[0]
#    std_devs = tf.sqrt(tf.linalg.diag_part(original_covariance))
#    modified_std_devs_diag = std_devs / tf.maximum(tf.constant(1.0, dtype=dtype), tf.constant(eps, dtype=dtype))
#    modified_std_devs_off_diag = std_devs * tf.maximum(tf.constant(0.0, dtype=dtype), tf.constant(1 - eps, dtype=dtype))
#    correlation_matrix = original_covariance / (std_devs[:, None] * std_devs[None, :])
#    modified_diag = tf.linalg.diag(modified_std_devs_diag**2)
#    outer_std_devs = modified_std_devs_off_diag[:, None] * modified_std_devs_off_diag[None, :]
#    modified_off_diag = correlation_matrix * outer_std_devs
#    diagonal_mask = tf.linalg.diag(tf.ones(shape, dtype=dtype))
#    modified_off_diag = modified_off_diag * (1 - diagonal_mask)
#    modified_covariance = modified_diag + modified_off_diag
#    return modified_covariance
#    
#def deform_cov_off_diag(d: Union[tfp.distributions.Distribution, tf.Tensor],
#                        eps: float = 0.,
#                        seed: int = 0,
#                        nsamples: int = 100_000
#                       ) -> Union[tfp.distributions.Distribution, tf.Tensor]:
#    if eps < 0:
#        raise ValueError("Epsilon must be non-negative")
#    if float(eps) == 0.:
#        return d
#    GMetrics.utils.reset_random_seeds(seed)
#    if isinstance(d, tf.Tensor):
#        original_mean = tf.reduce_mean(d,axis=0)
#        original_covariance = tfp.stats.covariance(d, sample_axis = 0)
#    elif isinstance(d, tfd.Distribution):
#        dtype = d.mean().dtype
#        original_mean = d.mean()
#        try:
#            original_covariance = d.covariance()
#        except:
#            samp = tf.cast(d.sample(nsamples), dtype = dtype)
#            original_covariance = tfp.stats.covariance(samp, sample_axis = 0)
#    else:
#        raise ValueError("Input must be either a tf.Tensor or a tfd.Distribution")
#    modified_covariance = modify_covariance_matrix(original_covariance, eps)
#    chol_original = tf.linalg.cholesky(original_covariance)
#    chol_modified = tf.linalg.cholesky(modified_covariance)
#    transformation_matrix_transpose = tf.linalg.triangular_solve(tf.linalg.matrix_transpose(chol_original), 
#                                                                 tf.linalg.matrix_transpose(chol_modified),
#                                                                 lower=False)
#    transformation_matrix = tf.linalg.matrix_transpose(transformation_matrix_transpose)
#    shift_to_zero = tfb.Shift(-original_mean)
#    linear_transform = tfb.ScaleMatvecTriL(scale_tril=transformation_matrix)
#    shift_back = tfb.Shift(original_mean)
#    chained_bijector = tfb.Chain([shift_back, linear_transform, shift_to_zero])
#    if isinstance(d, tf.Tensor):
#        deformed_d = chained_bijector.forward(d)
#    elif isinstance(d, tfd.Distribution):
#        deformed_d = tfd.TransformedDistribution(distribution = d,
#                                                 bijector = chained_bijector)
#    return deformed_d

def generate_partial_permutation(num_points: int, 
                                 indices_to_shuffle: tf.Tensor, 
                                 seed: int) -> tf.Tensor:
    tf.random.set_seed(seed)
    perm = tf.range(num_points)
    shuffled_indices = tf.random.shuffle(indices_to_shuffle)
    perm = tf.tensor_scatter_nd_update(perm, tf.expand_dims(indices_to_shuffle, 1), shuffled_indices)
    return perm

class DynamicShufflingBijector(tfb.Bijector):
    def __init__(self, 
                 fraction_shuffles: float, 
                 event_shape: int, 
                 seed: int, 
                 shuffle_type: str = "random", 
                 validate_args=False, name="DynamicShufflingBijector"):
        super(DynamicShufflingBijector, self).__init__(forward_min_event_ndims=1, validate_args=validate_args, name=name)
        self.fraction_shuffles = fraction_shuffles
        self.event_shape = event_shape
        self.seed = seed
        self.shuffle_type = shuffle_type

    def _get_indices_to_shuffle(self, num_points: int) -> tf.Tensor:
        num_shuffles = tf.cast(tf.cast(num_points, tf.float32) * tf.cast(self.fraction_shuffles, tf.float32), tf.int32) # type: ignore
        if self.shuffle_type == "random":
            indices_to_shuffle = tf.random.shuffle(tf.range(num_points))[:num_shuffles]
        elif self.shuffle_type == "firsts":
            indices_to_shuffle = tf.range(num_shuffles)
        elif self.shuffle_type == "lasts":
            indices_to_shuffle = tf.range(num_points - num_shuffles, num_points) # type: ignore
        else:
            raise ValueError("Shuffle must be one of 'random', 'firsts', or 'lasts'")
        return indices_to_shuffle

    def _forward(self, x) -> tf.Tensor:
        num_points = tf.shape(x)[0]
        shuffled_features = []
        
        for feature_idx in range(self.event_shape):
            indices_to_shuffle = self._get_indices_to_shuffle(num_points)
            perm = generate_partial_permutation(num_points = num_points, 
                                                indices_to_shuffle = indices_to_shuffle, 
                                                seed = self.seed + feature_idx)
            shuffled_feature = tf.gather(x[:, feature_idx], perm)
            shuffled_features.append(shuffled_feature)
        
        shuffled_x = tf.stack(shuffled_features, axis=1)
        return shuffled_x

    def _inverse(self, y) -> tf.Tensor:
        num_points = tf.shape(y)[0]
        shuffled_features = []
        
        for feature_idx in range(self.event_shape):
            indices_to_shuffle = self._get_indices_to_shuffle(num_points)
            perm = generate_partial_permutation(num_points = num_points, 
                                                indices_to_shuffle = indices_to_shuffle, 
                                                seed = self.seed + feature_idx)
            inverse_perm = tf.argsort(perm)
            shuffled_feature = tf.gather(y[:, feature_idx], inverse_perm)
            shuffled_features.append(shuffled_feature)
        
        shuffled_y = tf.stack(shuffled_features, axis=1)
        return shuffled_y

    def _forward_log_det_jacobian(self, x):
        return tf.zeros(tf.shape(x)[0], dtype=x.dtype)

def deform_cov_off_diag(d: Union[tfp.distributions.Distribution, tf.Tensor], # type: ignore
                        eps: float = 0., 
                        shuffle_type: str = "random", # allowed values: "random", "firsts", "lasts"
                        seed: int = 0
                       ) -> Union[tfp.distributions.Distribution, tf.Tensor]:
    if eps < 0:
        raise ValueError("Epsilon must be between 0 and 1")
    if float(eps) == 0.:
        return d
    
    GMetrics.utils.reset_random_seeds(seed)
    eps_off = np.minimum(eps, 1)
    eps_diag = np.maximum(eps - 1, 0)
    if eps_diag > 0:
        deformed_d = deform_cov_diag(d, eps = eps_diag, seed = seed)
    else:
        deformed_d = d
        
    if isinstance(d, tf.Tensor):
        event_shape = d.shape[1]
    elif isinstance(d, tfd.Distribution):
        event_shape = d.event_shape[0]
    deformation = DynamicShufflingBijector(fraction_shuffles = eps_off, 
                                           event_shape = event_shape, 
                                           shuffle_type = shuffle_type, 
                                           seed = seed)
    if isinstance(d, tf.Tensor):
        deformed_d = deformation.forward(deformed_d)
    elif isinstance(d, tfd.Distribution):
        deformed_d = tfd.TransformedDistribution(distribution = deformed_d,
                                                 bijector = deformation)
    
    return deformed_d

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

def deform_power_abs(d: Union[tfp.distributions.Distribution, tf.Tensor],
                     eps: float = 0.,
                     direction: str = "up"
                    ) -> Union[tfp.distributions.Distribution, tf.Tensor]:
    if eps < 0:
        raise ValueError("Epsilon must be non-negative")
    if float(eps) == 0.:
        return d
    if direction == "up":
        deformation = AbsPowerTransform(power = 1 + eps)
    elif direction == "down":
        deformation = AbsPowerTransform(power = 1 - eps)
    else:
         raise ValueError("Direction must be either 'up' or 'down'")
    if isinstance(d, tf.Tensor):
        deformed_d = deformation.forward(d)
    elif isinstance(d, tfd.Distribution):
        deformed_d = tfd.TransformedDistribution(distribution = d,
                                                 bijector = deformation)
    return deformed_d

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
            shift = tf.random.normal(shape = tf.shape(x), 
                                     stddev = self.scale, 
                                     dtype = x.dtype)
        elif self.shift_dist == "uniform":
            shift = tf.random.uniform(shape = tf.shape(x), 
                                      minval = -1 * self.scale, 
                                      maxval = self.scale, 
                                      dtype = x.dtype)
        return x + shift

    def _inverse(self, y):
        # This is not truly correct as we don't know the original shift
        raise NotImplementedError("Inverse is not well defined for random shifts.")

    def _forward_log_det_jacobian(self, x):
        # The log determinant of the Jacobian for a pure shift is zero
        return tf.zeros_like(x)
    
def deform_random(d: Union[tfp.distributions.Distribution, tf.Tensor],
                  eps: float = 0.,
                  shift_dist: str = "normal",
                  seed: int = 0
                 ) -> Union[tfp.distributions.Distribution, tf.Tensor]:
    if eps < 0:
        raise ValueError("Epsilon must be non-negative")
    if float(eps) == 0.:
        return d
    GMetrics.utils.reset_random_seeds(seed)
    deformation = RandomShift(scale = eps,
                              shift_dist = shift_dist)
    if isinstance(d, tf.Tensor):
        deformed_d = deformation.forward(d)
    elif isinstance(d, tfd.Distribution):
        deformed_d = tfd.TransformedDistribution(distribution = d,
                                                 bijector = deformation)
    return deformed_d

def deformed_distribution(d: Union[tfp.distributions.Distribution, tf.Tensor],
                          eps: float = 0.,
                          deform_type: str = "mean",
                          **deform_kwargs
                         ) -> Union[tfp.distributions.Distribution, tf.Tensor]:
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
    deformed_dist = func(d, eps = eps, **deform_kwargs)
    return deformed_dist