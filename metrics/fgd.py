__all__ = ["_linear",
           "_matrix_sqrtm",
           "_calculate_frechet_distance_tf",
           "_normalise_features_tf",
           "fgd_tf",
           "fgd_tf_fit",   
           "FGDMetric"]

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import traceback
from datetime import datetime
from timeit import default_timer as timer
from tqdm import tqdm # type: ignore
from scipy.stats import ks_2samp # type: ignore
from scipy.optimize import curve_fit # type: ignore
from .utils import reset_random_seeds
from .utils import conditional_print
from .utils import conditional_tf_print
from .utils import generate_and_clean_data
from .utils import NumpyDistribution
from .base import TwoSampleTestInputs
from .base import TwoSampleTestBase
from .base import TwoSampleTestResult
from .base import TwoSampleTestResults
from jetnet.evaluation import gen_metrics as JMetrics # type: ignore

from typing import Tuple, Union, Optional, Type, Dict, Any, List, Set
from .utils import DTypeType, IntTensor, FloatTensor, BoolTypeTF, BoolTypeNP, IntType, DataTypeTF, DataTypeNP, DataType, DistTypeTF, DistTypeNP, DistType, DataDistTypeNP, DataDistTypeTF, DataDistType, BoolType

def _linear(x, intercept, slope):
    return intercept + slope * x

@tf.function(jit_compile=False, reduce_retracing=True)
def _matrix_sqrtm(matrix: tf.Tensor) -> tf.Tensor:
    #with tf.device('/CPU:0'):
    sqrt_mat = tf.linalg.sqrtm(matrix)
    return sqrt_mat

@tf.function(jit_compile=False, reduce_retracing=True)
def _calculate_frechet_distance_tf(mu1_input: DataType,
                                   sigma1_input: DataType,
                                   mu2_input: DataType,
                                   sigma2_input: DataType,
                                   eps: float = 1e-6
                                  ) -> tf.Tensor:
    """
    TensorFlow implementation of the Frechet Distance.
    """
    mu1: tf.Tensor = tf.expand_dims(mu1_input, axis=-1) if mu1_input.shape.rank == 1 else tf.convert_to_tensor(mu1_input)
    mu2: tf.Tensor = tf.expand_dims(mu2_input, axis=-1) if mu2_input.shape.rank == 1 else tf.convert_to_tensor(mu2_input)
    sigma1 = tf.convert_to_tensor(sigma1_input, dtype=mu1.dtype)
    sigma2 = tf.convert_to_tensor(sigma2_input, dtype=mu2.dtype)

    diff: tf.Tensor = mu1 - mu2

    # Product might be almost singular
    covmean_sqrtm: tf.Tensor = _matrix_sqrtm(tf.linalg.matmul(sigma1, sigma2)) # type: ignore
    
    # Use tf.cond to handle possible numerical errors
    def adjust_covmean():
        offset: tf.Tensor = tf.cast(tf.eye(tf.shape(sigma1)[0]) * eps, sigma1.dtype)
        return _matrix_sqrtm(tf.linalg.matmul(sigma1 + offset, sigma2 + offset))

    covmean_sqrtm = tf.cond(tf.reduce_all(tf.math.is_finite(covmean_sqrtm)), 
                            lambda: covmean_sqrtm, 
                            adjust_covmean)

    # Use tf.cond to handle possible imaginary component
    covmean_sqrtm = tf.cond(tf.reduce_any(tf.math.imag(covmean_sqrtm) != 0),
                            lambda: tf.math.real(covmean_sqrtm),
                            lambda: covmean_sqrtm)

    tr_covmean: tf.Tensor = tf.linalg.trace(covmean_sqrtm)

    frechet_distance: tf.Tensor = tf.reduce_sum(diff * diff) + tf.linalg.trace(sigma1) + tf.linalg.trace(sigma2) - 2.0 * tr_covmean

    return frechet_distance

@tf.function(jit_compile = True, reduce_retracing = True)
def _normalise_features_tf(data1_input: DataType, 
                           data2_input: Optional[DataType] = None
                          ) -> Union[DataTypeTF, Tuple[DataTypeTF, DataTypeTF]]:
    data1: DataTypeTF = tf.convert_to_tensor(data1_input)
    maxes: tf.Tensor = tf.reduce_max(tf.abs(data1), axis=0)
    maxes = tf.where(tf.equal(maxes, 0), tf.ones_like(maxes), maxes)  # don't normalize in case of features which are just 0

    if data2_input is not None:
        data2: DataTypeTF = tf.convert_to_tensor(data2_input)
        return data1 / maxes, data2 / maxes
    else:
        return data1 / maxes

@tf.function(jit_compile=True)
def generate_unique_indices(num_samples, batch_size, num_batches, seed = None):
    reset_random_seeds(seed)
    # Edge case: if num_samples equals batch_size, shuffle uniquely for each batch
    if num_samples == batch_size:
        # Create a large tensor that repeats the range [0, num_samples] num_batches times
        indices = tf.tile(tf.range(num_samples, dtype=tf.int32)[tf.newaxis, :], [num_batches, 1])
        # Shuffle each batch's indices uniquely
        batched_indices = tf.map_fn(lambda x: tf.random.shuffle(x, seed = seed), indices, dtype=tf.int32)
    else:
        # Standard case handling (repeat shuffling logic you need for num_samples != batch_size)
        full_indices = tf.range(num_samples, dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(full_indices, seed = seed)
        if batch_size * num_batches > num_samples:
            multiples = (batch_size * num_batches // num_samples) + 1
            shuffled_indices = tf.tile(shuffled_indices, [multiples])
        batched_indices = tf.reshape(shuffled_indices[:batch_size * num_batches], [num_batches, batch_size])
    
    return batched_indices

@tf.function(jit_compile=False)
def fgd_tf(X: tf.Tensor, 
           Y: tf.Tensor,
           min_samples_input: Optional[int] = None, 
           max_samples_input: Optional[int] = None,
           num_batches: int = 20, 
           num_points: int = 10,
           normalise: bool = False,
           seed: int = 0
          ) -> Tuple[DataTypeTF, tf.Tensor]:
    # if normalise is True, normalise the features
    normalise = tf.constant(normalise, dtype=tf.bool)
    X = tf.cond(normalise, lambda: _normalise_features_tf(X), lambda: X)
    Y = tf.cond(normalise, lambda: _normalise_features_tf(Y), lambda: Y)
    dtype = X.dtype
    
    nsamplesX: int = len(X)
    nsamplesY: int = len(Y)
    
    min_samples: int = min_samples_input if min_samples_input is not None else int(min(nsamplesX, nsamplesY) / 3)
    max_samples: int = max_samples_input if max_samples_input is not None else int(min(nsamplesX, nsamplesY))

    if len(X.shape) == 1: # type: ignore
        X = tf.expand_dims(X, axis=-1)
    if len(Y.shape) == 1: # type: ignore
        Y = tf.expand_dims(Y, axis=-1)
        
    # Preallocate random indices
    max_batch_size: int = max_samples
    total_batches: int = num_points * num_batches
    #all_rand1: tf.Tensor = tf.random.uniform(shape=[total_batches, max_batch_size], minval=0, maxval=tf.shape(X)[0], dtype=tf.int32, seed=seed)
    #all_rand2: tf.Tensor = tf.random.uniform(shape=[total_batches, max_batch_size], minval=0, maxval=tf.shape(Y)[0], dtype=tf.int32, seed=seed)
    #all_rand1: tf.Tensor = generate_unique_indices(nsamplesX, max_batch_size, total_batches, seed) # type: ignore
    #all_rand2: tf.Tensor = generate_unique_indices(nsamplesY, max_batch_size, total_batches, (seed + 10000) * 1000) # type: ignore
    batches: tf.Tensor = tf.cast(1 / tf.linspace(1.0 / min_samples, 1.0 / max_samples, num_points), dtype=tf.int32) # type: ignore
        
    vals = tf.TensorArray(dtype, size=num_points, clear_after_read=False)
    def body_outer(i, vals):
        batch_size: tf.Tensor = batches[i] # type: ignore
        val_points: tf.TensorArray = tf.TensorArray(dtype, size=num_batches)
        def body_inner(j, val_points):
            start_index: tf.Tensor = i * num_batches + j
            #rand1: tf.Tensor = all_rand1[start_index, :batch_size] # type: ignore
            #rand2: tf.Tensor = all_rand2[start_index, :batch_size] # type: ignore
            #rand_sample1: tf.Tensor = tf.gather(X, rand1)
            #rand_sample2: tf.Tensor = tf.gather(Y, rand2)
            rand_sample1: tf.Tensor = tf.random.shuffle(X)[:batch_size]
            rand_sample2: tf.Tensor = tf.random.shuffle(Y)[:batch_size]
            mu1: tf.Tensor = tf.reduce_mean(rand_sample1, axis=0)
            mu2: tf.Tensor = tf.reduce_mean(rand_sample2, axis=0)
            sigma1: tf.Tensor = tfp.stats.covariance(rand_sample1, sample_axis=0, event_axis=-1)
            sigma2: tf.Tensor = tfp.stats.covariance(rand_sample2, sample_axis=0, event_axis=-1)
            val: tf.Tensor = _calculate_frechet_distance_tf(mu1, sigma1, mu2, sigma2) # type: ignore
            val_points = val_points.write(j, val)
            return j + 1, val_points
        _, val_points = tf.while_loop(
            cond=lambda j, _: j < num_batches,
            body=body_inner,
            loop_vars=(0, val_points)
        )
        mean_val: tf.Tensor = tf.reduce_mean(val_points.stack())
        vals = vals.write(i, mean_val)
        return i + 1, vals
    _, vals = tf.while_loop(
        cond=lambda i, _: i < num_points,
        body=body_outer,
        loop_vars=(0, vals)
    )
    vals_stacked: DataTypeTF = vals.stack()
    
    #vals: tf.TensorArray = tf.TensorArray(dtype = X.dtype, size = num_points)
    #counter: int = 0
    #for i in tf.range(tf.shape(batches)[0]):
    #    batch_size: int = batches[i]
    #    val_points: tf.TensorArray = tf.TensorArray(dtype = X.dtype, size = num_batches)
    #    for j in tf.range(num_batches):
    #        rand1: tf.Tensor = all_rand1[counter, :batch_size]
    #        rand2: tf.Tensor = all_rand2[counter, :batch_size]
    #        counter += 1
    #        rand_sample1: tf.Tensor = tf.gather(X, rand1)
    #        rand_sample2: tf.Tensor = tf.gather(Y, rand2)
    #        mu1: tf.Tensor = tf.reduce_mean(rand_sample1, axis=0)
    #        mu2: tf.Tensor = tf.reduce_mean(rand_sample2, axis=0)
    #        sigma1: tf.Tensor = tfp.stats.covariance(rand_sample1, sample_axis=0, event_axis=-1)
    #        sigma2: tf.Tensor = tfp.stats.covariance(rand_sample2, sample_axis=0, event_axis=-1)
    #        val: tf.Tensor = _calculate_frechet_distance_tf(mu1, sigma1, mu2, sigma2) # type: ignore
    #        val_points = val_points.write(j, val)
    #    val_points = val_points.stack()
    #    vals = vals.write(i, tf.reduce_mean(val_points))
    #vals_stacked: DataTypeTF = vals.stack()
    
    return vals_stacked, batches

def fgd_tf_fit(vals_list_input: DataType, 
               batches_list_input: tf.Tensor
              ) -> Tuple[DataTypeNP, DataTypeNP]:
    if len(tf.shape(vals_list_input)) == 1:
        vals_list_input = tf.expand_dims(vals_list_input, axis=0)
    if len(tf.shape(batches_list_input)) == 1:
        batches_list_input = tf.expand_dims(batches_list_input, axis=0)
    vals_list: DataTypeNP = np.array(vals_list_input)
    batches_list: DataTypeNP = np.array(batches_list_input)
    metric_list: list = []
    metric_error_list: list = []

    for vals, batches in zip(vals_list, batches_list):
        try:
            # Using curve_fit with a try-except block
            params, covs = curve_fit(_linear, 1 / batches, vals, bounds=([0, 0], [np.inf, np.inf]))
            metric = params[0]
            metric_error = np.sqrt(np.diag(covs)[0])
        except Exception as e:
            # Handle the exception by propagating NaN or some other indicator
            metric = np.nan
            metric_error = np.nan

        metric_list.append(metric)
        metric_error_list.append(metric_error)

    return np.array(metric_list), np.array(metric_error_list)

class FGDMetric(TwoSampleTestBase):
    """
    """
    def __init__(self, 
                 data_input: TwoSampleTestInputs,
                 progress_bar: bool = False,
                 verbose: bool = False,
                 **fgd_kwargs
                ) -> None:
        # From base class
        self._Inputs: TwoSampleTestInputs
        self._progress_bar: bool
        self._verbose: bool
        self._start: float
        self._end: float
        self._pbar: tqdm
        self._Results: TwoSampleTestResults
        
        # New attributes
        self.fgd_kwargs = fgd_kwargs # Use the setter to validate arguments
        
        super().__init__(data_input = data_input, 
                         progress_bar = progress_bar,
                         verbose = verbose)
        
    @property
    def fgd_kwargs(self) -> Dict[str, Any]:
        return self._fgd_kwargs
    
    @fgd_kwargs.setter
    def fgd_kwargs(self, fgd_kwargs: Dict[str, Any]) -> None:
        valid_keys: Set[str] = {'min_samples_input', 'max_samples_input', 'num_batches', 'num_points', 'normalise', 'seed'}
        # Dynamically get valid keys from `fgd` function's parameters
        # valid_keys = set(inspect.signature(JMetrics.fgd).parameters.keys())
        
        for key in fgd_kwargs.keys():
            if key not in valid_keys:
                raise ValueError(f"Invalid key: {key}. Valid keys are {valid_keys}")

        # You can add more specific validations for each argument here
        if 'min_samples' in fgd_kwargs and (not isinstance(fgd_kwargs['min_samples'], int) or fgd_kwargs['min_samples'] <= 0):
            raise ValueError("min_samples must be a positive integer")
            
        if 'max_samples' in fgd_kwargs and (not isinstance(fgd_kwargs['max_samples'], int) or fgd_kwargs['max_samples'] <= 0):
            raise ValueError("max_samples must be a positive integer")
        
        if 'num_batches' in fgd_kwargs and (not isinstance(fgd_kwargs['num_batches'], int) or fgd_kwargs['num_batches'] <= 0):
            raise ValueError("num_batches must be a positive integer")
            
        if 'num_points' in fgd_kwargs and (not isinstance(fgd_kwargs['num_points'], int) or fgd_kwargs['num_points'] <= 0):
            raise ValueError("num_points must be a positive integer")
        
        if 'normalise' in fgd_kwargs and not isinstance(fgd_kwargs['normalise'], bool):
            raise ValueError("normalise must be a boolean")
        
        if 'seed' in fgd_kwargs and not isinstance(fgd_kwargs['seed'], int):
            raise ValueError("seed must be an integer")
        
        self._fgd_kwargs = fgd_kwargs
        
            
    def compute(self, max_vectorize: int = 100) -> None:
        """
        Function that computes the FGD  metric (and its uncertainty) from two multivariate samples
        selecting among the Test_np and Test_tf methods depending on the value of the use_tf attribute.
        
        Parameters:
        ----------
        max_vectorize: int, optional, default = 100
            Maximum number of samples that can be processed by the tensorflow backend.
            If None, the total number of samples is not checked.

        Returns:
        -------
        None
        """
        if self.use_tf:
            self.Test_tf(max_vectorize = max_vectorize)
        else:
            self.Test_np()
    
    def Test_np(self) -> None:
        """
        """
        # Set alias for inputs
        if isinstance(self.Inputs.dist_1_num, np.ndarray):
            dist_1_num: DataTypeNP = self.Inputs.dist_1_num
        else:
            dist_1_num = self.Inputs.dist_1_num.numpy()
        if isinstance(self.Inputs.dist_2_num, np.ndarray):
            dist_2_num: DataTypeNP = self.Inputs.dist_2_num
        else:
            dist_2_num = self.Inputs.dist_2_num.numpy()
        dist_1_symb: DistType = self.Inputs.dist_1_symb
        dist_2_symb: DistType = self.Inputs.dist_2_symb
        ndims: int = self.Inputs.ndims
        niter: int
        batch_size: int
        niter, batch_size = self.get_niter_batch_size_np() # type: ignore
        if isinstance(self.Inputs.dtype, tf.DType):
            dtype: Union[type, np.dtype] = self.Inputs.dtype.as_numpy_dtype
        else:
            dtype = self.Inputs.dtype
        dist_1_k: DataTypeNP
        dist_2_k: DataTypeNP
        
        # Utility functions
        def set_dist_num_from_symb(dist: DistType,
                                   nsamples: int,
                                   dtype: Union[type, np.dtype],
                                  ) -> DataTypeNP:
            if isinstance(dist, tfp.distributions.Distribution):
                dist_num_tmp: DataTypeTF = generate_and_clean_data(dist, nsamples, self.Inputs.batch_size_gen, dtype = dtype, seed_generator = self.Inputs.seed_generator, strategy = self.Inputs.strategy) # type: ignore
                dist_num: DataTypeNP = dist_num_tmp.numpy().astype(dtype) # type: ignore
            elif isinstance(dist, NumpyDistribution):
                dist_num = dist.sample(nsamples).astype(dtype = dtype)
            else:
                raise TypeError("dist must be either a tfp.distributions.Distribution or a NumpyDistribution object.")
            return dist_num
        
        def start_calculation() -> None:
            conditional_print(self.verbose, "\n------------------------------------------")
            conditional_print(self.verbose, "Starting FGD metric calculation...")
            conditional_print(self.verbose, "niter = {}" .format(niter))
            conditional_print(self.verbose, "batch_size = {}" .format(batch_size))
            self._start = timer()
            
        def init_progress_bar() -> None:
            nonlocal niter
            if self.progress_bar:
                self.pbar = tqdm(total = niter, desc="Iterations")

        def update_progress_bar() -> None:
            if not self.pbar.disable:
                self.pbar.update(1)

        def close_progress_bar() -> None:
            if not self.pbar.disable:
                self.pbar.close()

        def end_calculation() -> None:
            self._end = timer()
            conditional_print(self.verbose, "Two-sample test calculation completed in "+str(self.end-self.start)+" seconds.")
        
        metric_list: List[float] = []
        metric_error_list: List[float] = []

        start_calculation()
        init_progress_bar()
            
        self.Inputs.reset_seed_generator()
        
        conditional_print(self.verbose, "Running numpy FGD calculation...")
        for k in range(niter):
            if not np.shape(dist_1_num[0])[0] == 0 and not np.shape(dist_2_num[0])[0] == 0:
                dist_1_k = dist_1_num[k*batch_size:(k+1)*batch_size,:]
                dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            elif not np.shape(dist_1_num[0])[0] == 0 and np.shape(dist_2_num[0])[0] == 0:
                dist_1_k = dist_1_num[k*batch_size:(k+1)*batch_size,:]
                dist_2_k = set_dist_num_from_symb(dist = dist_2_symb, nsamples = batch_size, dtype = dtype)
            elif np.shape(dist_1_num[0])[0] == 0 and not np.shape(dist_2_num[0])[0] == 0:
                dist_1_k = set_dist_num_from_symb(dist = dist_1_symb, nsamples = batch_size, dtype = dtype)
                dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            else:
                dist_1_k = set_dist_num_from_symb(dist = dist_1_symb, nsamples = batch_size, dtype = dtype)
                dist_2_k = set_dist_num_from_symb(dist = dist_2_symb, nsamples = batch_size, dtype = dtype)
            metric: float
            metric_error: float
            metric, metric_error = JMetrics.fgd(dist_1_k, dist_2_k, **self.fgd_kwargs)
            metric_list.append(metric)
            metric_error_list.append(metric_error)
            update_progress_bar()
        
        close_progress_bar()
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "FGD Test_np"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "numpy"}}
        result_value: Dict[str, Optional[DataTypeNP]] = {"metric_list": np.array(metric_list),
                                                         "metric_error_list": np.array(metric_error_list)}
        result: TwoSampleTestResult = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
        self.Results.append(result)
        
    def Test_tf(self, max_vectorize: int = 100) -> None:
        """
        Function that computes the FGD  metric (and its uncertainty) from two multivariate samples
        using tensorflow functions.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The calculation is parallelized over max_vectorize (out of niter).
        The results are stored in the Results attribute.

        Parameters:
        ----------
        max_vectorize: int, optional, default = 100
            A maximum number of batch_size*max_vectorize samples per time are processed by the tensorflow backend.
            Given a value of max_vectorize, the niter FGD calculations are split in chunks of max_vectorize.
            Each chunk is processed by the tensorflow backend in parallel. If ndims is larger than max_vectorize,
            the calculation is vectorized niter times over ndims.

        Returns:
        --------
        None
        """
        max_vectorize = int(max_vectorize)
        # Set alias for inputs
        if isinstance(self.Inputs.dist_1_num, np.ndarray):
            dist_1_num: tf.Tensor = tf.convert_to_tensor(self.Inputs.dist_1_num)
        else:
            dist_1_num = self.Inputs.dist_1_num # type: ignore
        if isinstance(self.Inputs.dist_2_num, np.ndarray):
            dist_2_num: tf.Tensor = tf.convert_to_tensor(self.Inputs.dist_2_num)
        else:
            dist_2_num = self.Inputs.dist_2_num # type: ignore
        if isinstance(self.Inputs.dist_1_symb, tfp.distributions.Distribution):
            dist_1_symb: tfp.distributions.Distribution = self.Inputs.dist_1_symb
        else:
            raise TypeError("dist_1_symb must be a tfp.distributions.Distribution object when use_tf is True.")
        if isinstance(self.Inputs.dist_2_symb, tfp.distributions.Distribution):
            dist_2_symb: tfp.distributions.Distribution = self.Inputs.dist_2_symb
        else:
            raise TypeError("dist_2_symb must be a tfp.distributions.Distribution object when use_tf is True.")
        ndims: int = self.Inputs.ndims
        niter: int
        batch_size: int
        niter, batch_size = [int(i) for i in self.get_niter_batch_size_tf()] # type: ignore
        dtype: tf.DType = tf.as_dtype(self.Inputs.dtype)
        
        # Utility functions
        def start_calculation() -> None:
            conditional_tf_print(self.verbose, "\n------------------------------------------")
            conditional_tf_print(self.verbose, "Starting FGD metric calculation...")
            conditional_tf_print(self.verbose, "Running TF FGD calculation...")
            conditional_tf_print(self.verbose, "niter =", niter)
            conditional_tf_print(self.verbose, "batch_size =", batch_size)
            self._start = timer()

        def end_calculation() -> None:
            self._end = timer()
            elapsed = self.end - self.start
            conditional_tf_print(self.verbose, "FGD metric calculation completed in", str(elapsed), "seconds.")
                    
        def set_dist_num_from_symb(dist: DistTypeTF,
                                   nsamples: int,
                                  ) -> tf.Tensor:
            dist_num: tf.Tensor = generate_and_clean_data(dist, nsamples, self.Inputs.batch_size_gen, dtype = self.Inputs.dtype, seed_generator = self.Inputs.seed_generator, strategy = self.Inputs.strategy) # type: ignore
            return dist_num
        
        def return_dist_num(dist_num: tf.Tensor) -> tf.Tensor:
            return dist_num
        
        @tf.function(jit_compile=False, reduce_retracing=True)
        def batched_test_sub(dist_1_k_replica: tf.Tensor, 
                             dist_2_k_replica: tf.Tensor
                            ) -> DataTypeTF:
            def loop_body(idx):
                vals, batches = fgd_tf(dist_1_k_replica[idx, :, :], dist_2_k_replica[idx, :, :], **self.fgd_kwargs) # type: ignore
                vals = tf.cast(vals, dtype=dtype)
                batches = tf.cast(batches, dtype=dtype)
                return vals, batches

            # Vectorize over ndims*chunk_size
            vals_list: tf.Tensor
            batches_list: tf.Tensor
            vals_list, batches_list = tf.vectorized_map(loop_body, tf.range(tf.shape(dist_1_k_replica)[0])) # type: ignore
            
            res: DataTypeTF = tf.concat([vals_list, batches_list], axis=1) # type: ignore
            return res
        
        #@tf.function(jit_compile=False, reduce_retracing=True)
        def batched_test(start: tf.Tensor, 
                         end: tf.Tensor
                        ) -> DataTypeTF:
            # Define batched distributions
            dist_1_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_1_symb, nsamples = batch_size * (end - start)), # type: ignore
                                               false_fn = lambda: return_dist_num(dist_1_num[start * batch_size : end * batch_size, :])) # type: ignore
            dist_2_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_2_symb, nsamples = batch_size * (end - start)), # type: ignore
                                               false_fn = lambda: return_dist_num(dist_2_num[start * batch_size : end * batch_size, :])) # type: ignore

            dist_1_k = tf.reshape(dist_1_k, (end - start, batch_size, ndims)) # type: ignore
            dist_2_k = tf.reshape(dist_2_k, (end - start, batch_size, ndims)) # type: ignore

            res: DataTypeTF = batched_test_sub(dist_1_k, dist_2_k) # type: ignore
    
            return res
        
        def compute_test(max_vectorize: int = 100) -> Tuple[DataTypeTF, tf.Tensor]:
            # Check if numerical distributions are empty and print a warning if so
            conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_1_num tensor is empty. Batches will be generated 'on-the-fly' from dist_1_symb.") # type: ignore
            conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_2_num tensor is empty. Batches will be generated 'on-the-fly' from dist_2_symb.") # type: ignore
            
            # Ensure that max_vectorize is an integer larger than ndims
            max_vectorize = int(tf.cast(tf.minimum(max_vectorize, niter),tf.int32)) # type: ignore

            # Compute the maximum number of iterations per chunk
            max_iter_per_chunk: int = max_vectorize # type: ignore
            
            # Compute the number of chunks
            nchunks: int = int(tf.cast(tf.math.ceil(niter / max_iter_per_chunk), tf.int32)) # type: ignore
            conditional_tf_print(tf.logical_and(self.verbose,tf.logical_not(tf.equal(nchunks,1))), "nchunks =", nchunks) # type: ignore

            res: tf.TensorArray = tf.TensorArray(dtype, size = nchunks)
            res_vals: tf.TensorArray = tf.TensorArray(dtype, size = nchunks)
            res_batches: tf.TensorArray = tf.TensorArray(dtype, size = nchunks)

            def body(i: int, 
                     res: tf.TensorArray
                    ) -> Tuple[int, tf.TensorArray]:
                start: tf.Tensor = tf.cast(i * max_iter_per_chunk, tf.int32) # type: ignore
                end: tf.Tensor = tf.cast(tf.minimum(start + max_iter_per_chunk, niter), tf.int32) # type: ignore
                conditional_tf_print(tf.logical_and(tf.logical_or(tf.math.logical_not(tf.equal(start,0)),tf.math.logical_not(tf.equal(end,niter))), self.verbose), "Iterating from", start, "to", end, "out of", niter, ".") # type: ignore
                chunk_result: DataTypeTF = batched_test(start, end) # type: ignore
                res = res.write(i, chunk_result)
                return i+1, res
            
            def cond(i: int, 
                     res: tf.TensorArray):
                return i < nchunks
            
            _, res = tf.while_loop(cond, body, [0, res])
            
            for i in range(nchunks):
                res_i: DataTypeTF = tf.convert_to_tensor(res.read(i))
                npoints: tf.Tensor = res_i.shape[1] // 2 # type: ignore
                res_vals = res_vals.write(i, res_i[:, :npoints]) # type: ignore
                res_batches = res_batches.write(i, res_i[:, npoints:]) # type: ignore
                
            vals_list: DataTypeTF = res_vals.stack() # type: ignore
            batches_list: tf.Tensor = res_batches.stack() # type: ignore
            
            shape = tf.shape(vals_list)
            vals_list = tf.reshape(vals_list, (shape[0] * shape[1], shape[2]))
            batches_list = tf.reshape(batches_list, (shape[0] * shape[1], shape[2]))
            #vals_list: DataTypeTF = tf.squeeze(res_vals.stack())
            #batches_list: tf.Tensor = tf.squeeze(res_batches.stack())
            
            # Flatten vals_list and batches_list to 1-D arrays
            #vals_list = tf.reshape(vals_list, [-1])  # Flatten to 1-D
            #batches_list = tf.reshape(batches_list, [-1])  # Flatten to 1-D

            return vals_list, batches_list

        start_calculation()
        
        self.Inputs.reset_seed_generator()
        
        vals_list: DataTypeTF
        batches_list: tf.Tensor
        vals_list, batches_list  = compute_test(max_vectorize = max_vectorize)
                
        #print(f"vals_list: {vals_list=}")
        #print(f"batches_list: {batches_list=}")
        
        metric_list: DataTypeNP
        metric_error_list: DataTypeNP
        metric_list, metric_error_list = fgd_tf_fit(vals_list, batches_list)
                             
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "FGD Test_tf"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "tensorflow"}}
        result_value: Dict[str, Optional[DataTypeNP]] = {"metric_list": metric_list,
                                                         "metric_error_list": metric_error_list}
        result: TwoSampleTestResult = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
        self.Results.append(result)