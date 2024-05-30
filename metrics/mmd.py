__all__ = ["_poly_kernel_pairwise_tf",
           "_mmd_quadratic_unbiased_tf",
           "_mmd_poly_quadratic_unbiased_tf",
           "_mmd_batches_tf",
           "mmd_tf",
           "MMDMetric"]

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import traceback
from datetime import datetime
from timeit import default_timer as timer
from tqdm import tqdm # type: ignore
from scipy.stats import iqr # type: ignore
from scipy.optimize import curve_fit # type: ignore
from GMetrics.utils import reset_random_seeds
from GMetrics.utils import conditional_print
from GMetrics.utils import conditional_tf_print
from GMetrics.utils import generate_and_clean_data
from GMetrics.utils import NumpyDistribution
from GMetrics.base import TwoSampleTestInputs
from GMetrics.base import TwoSampleTestBase
from GMetrics.base import TwoSampleTestResult
from GMetrics.base import TwoSampleTestResults
from jetnet.evaluation import gen_metrics as JMetrics # type: ignore

from typing import Tuple, Union, Optional, Type, Dict, Any, List, Set
from GMetrics.utils import DTypeType, IntTensor, FloatTensor, BoolTypeTF, BoolTypeNP, IntType, DataTypeTF, DataTypeNP, DataType, DistTypeTF, DistTypeNP, DistType, DataDistTypeNP, DataDistTypeTF, DataDistType, BoolType

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

@tf.function(jit_compile=True, reduce_retracing = True)
def _poly_kernel_pairwise_tf(X, Y, degree):
    gamma = tf.cast(1.0, X.dtype) / tf.cast(tf.shape(X)[-1], X.dtype)
    return tf.pow(tf.linalg.matmul(X, Y, transpose_b=True) * gamma + 1.0, degree)

@tf.function(jit_compile=True, reduce_retracing = True)
def _mmd_quadratic_unbiased_tf(XX, YY, XY):
    m = tf.cast(tf.shape(XX)[0], XX.dtype)
    n = tf.cast(tf.shape(YY)[0], YY.dtype)
    return (tf.reduce_sum(XX) - tf.linalg.trace(XX)) / (m * (m - 1)) \
           + (tf.reduce_sum(YY) - tf.linalg.trace(YY)) / (n * (n - 1)) \
           - 2 * tf.reduce_mean(XY)
           
@tf.function(jit_compile=True, reduce_retracing = True)
def _mmd_poly_quadratic_unbiased_tf(X, Y, degree=4):
    XX = _poly_kernel_pairwise_tf(X, X, degree=degree)
    YY = _poly_kernel_pairwise_tf(Y, Y, degree=degree)
    XY = _poly_kernel_pairwise_tf(X, Y, degree=degree)
    return _mmd_quadratic_unbiased_tf(XX, YY, XY)

@tf.function(jit_compile=True, reduce_retracing = True)
def _blockwize_mmd_poly_quadratic_unbiased_tf(X, Y, degree=4, block_size=10000):
    block_size = tf.cast(block_size, tf.int32)
    num_samples_X = tf.shape(X)[0]
    num_samples_Y = tf.shape(Y)[0]
    num_blocks_X = tf.math.ceil(tf.cast(num_samples_X, tf.float32) / tf.cast(block_size, tf.float32))
    num_blocks_X = tf.cast(num_blocks_X, tf.int32)
    num_blocks_Y = tf.math.ceil(tf.cast(num_samples_Y, tf.float32) / tf.cast(block_size, tf.float32))
    num_blocks_Y = tf.cast(num_blocks_Y, tf.int32)
    mmd_result = tf.constant(0.0, dtype=X.dtype)
    m = tf.cast(tf.shape(X)[0], X.dtype)
    n = tf.cast(tf.shape(Y)[0], Y.dtype)
    
    def loop_body(i, mmd_result):
        start_i = i * block_size
        end_i = tf.minimum(start_i + block_size, num_samples_X)

        def inner_loop_body(j, mmd_result):
            start_j = j * block_size
            end_j = tf.minimum(start_j + block_size, num_samples_Y)
            X_block_i = X[start_i:end_i]
            X_block_j = X[start_j:end_j]
            Y_block_i = Y[start_i:end_i]
            Y_block_j = Y[start_j:end_j]
            XX_block = _poly_kernel_pairwise_tf(X_block_i, X_block_j, degree)
            YY_block = _poly_kernel_pairwise_tf(Y_block_i, Y_block_j, degree)
            XY_block = _poly_kernel_pairwise_tf(X_block_i, Y_block_j, degree)

            trace_XX = tf.cond(tf.equal(i, j), lambda: tf.linalg.trace(XX_block), lambda: tf.cast(0.0,XX_block.dtype))
            trace_YY = tf.cond(tf.equal(i, j), lambda: tf.linalg.trace(YY_block), lambda: tf.cast(0.0,YY_block.dtype))

            block_mmd = (tf.reduce_sum(XX_block) - trace_XX) / (m * (m - 1)) \
                      + (tf.reduce_sum(YY_block) - trace_YY) / (n * (n - 1)) \
                      - 2 * tf.reduce_sum(XY_block) / (m * n)

            return j + 1, mmd_result + block_mmd

        _, mmd_result = tf.while_loop(
            lambda j, _: j < num_blocks_Y,
            inner_loop_body,
            [0, mmd_result],
            parallel_iterations=10
        )
        return i + 1, mmd_result

    _, mmd_result = tf.while_loop(
        lambda i, _: i < num_blocks_X,
        loop_body,
        [0, mmd_result],
        parallel_iterations=10
    )
    
    return mmd_result

@tf.function(jit_compile=True, reduce_retracing = True)
def mmd_tf(X: tf.Tensor,
           Y: tf.Tensor,
           degree: int = 4,
           block_size: int = 10_000,
           normalise: bool = False,
           seed: int = 42):
    # if normalise is True, normalise the features
    normalise = tf.constant(normalise, dtype=tf.bool)
    X = tf.cond(normalise, lambda: _normalise_features_tf(X), lambda: X)
    Y = tf.cond(normalise, lambda: _normalise_features_tf(Y), lambda: Y)
    
    val = _blockwize_mmd_poly_quadratic_unbiased_tf(X, Y, degree, block_size)
    
    return val

#@tf.function(jit_compile=True, reduce_retracing = True)
#def mmd_tf(X: tf.Tensor,
#           Y: tf.Tensor,
#           degree: int = 4,
#           block_size: int = 10_000,
#           num_batches: int = 1,
#           batch_size: int = 10_000,
#           normalise: bool = False,
#           seed: int = 42):
#    if normalise:
#        X, Y = _normalise_features_tf(X, Y) # type: ignore
#    else:
#        X = tf.convert_to_tensor(Y)
#        X = tf.convert_to_tensor(Y)
#        
#    num_samples_X = tf.shape(X)[0]
#    num_samples_Y = tf.shape(Y)[0]
#    
#    vals_point = []
#    for i in range(num_batches):
#        selected_indices_X = generate_unique_indices(num_samples_X, batch_size, seed=seed + i)
#        selected_indices_Y = generate_unique_indices(num_samples_Y, batch_size, seed=seed + i * 2)
#        
#        rand_sample1 = tf.gather(X, selected_indices_X)
#        rand_sample2 = tf.gather(Y, selected_indices_Y)
#
#        val = _blockwize_mmd_poly_quadratic_unbiased_tf(rand_sample1, rand_sample2, degree, block_size)
#        #val = _mmd_poly_quadratic_unbiased_tf(rand_sample1, rand_sample2, degree)
#        vals_point.append(val)
#    
#    vals_point = tf.stack(vals_point)
#    return vals_point

def mmd_tf_output(vals_points_input: DataTypeTF) -> DataTypeTF:
    vals_points: DistTypeNP = np.array(vals_points_input)
    #print(f"Number of values computed: {len(vals_points)}")
    #print(f"vals_points is {vals_points}")
    metric_list: list = []
    metric_error_list: list = []
    if len(vals_points.shape) == 1:
        #print("Qui")
        metric_list.append(np.median(vals_points))
        try:
            metric_error_list.append(iqr(vals_points, rng=(16.275, 83.725)) / 2)
        except:
            metric_error_list.append(None)
    else:
        for vals_point in vals_points:
            #print("Qua")
            metric_list.append(np.median(vals_point))
            try:
                metric_error_list.append(iqr(vals_point, rng=(16.275, 83.725)) / 2)
            except:
                metric_error_list.append(None)
    return np.array(metric_list), np.array(metric_error_list)
           

class MMDMetric(TwoSampleTestBase):
    """
    """
    def __init__(self, 
                 data_input: TwoSampleTestInputs,
                 progress_bar: bool = False,
                 verbose: bool = False,
                 **mmd_kwargs
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
        self.mmd_kwargs = mmd_kwargs # Use the setter to validate arguments
        
        super().__init__(data_input = data_input, 
                         progress_bar = progress_bar,
                         verbose = verbose)
        
    @property
    def mmd_kwargs(self) -> Dict[str, Any]:
        return self._mmd_kwargs
    
    @mmd_kwargs.setter
    def mmd_kwargs(self, mmd_kwargs: Dict[str, Any]) -> None:
        valid_keys: Set[str] = {'degree', 'block_size', 'normalise', 'seed'}
        # Dynamically get valid keys from `mmd` function's parameters
        # valid_keys = set(inspect.signature(JMetrics.mmd).parameters.keys())
        
        for key in mmd_kwargs.keys():
            if key not in valid_keys:
                raise ValueError(f"Invalid key: {key}. Valid keys are {valid_keys}")
            
        if 'degree' in mmd_kwargs and not isinstance(mmd_kwargs['degree'], int):
            raise ValueError("degree must be an integer")
        
        if 'block_size' in mmd_kwargs and not isinstance(mmd_kwargs['block_size'], int):
            raise ValueError("block_size must be an integer")
        
        if 'normalise' in mmd_kwargs and not isinstance(mmd_kwargs['normalise'], bool):
            raise ValueError("normalise must be a boolean")
        
        if 'seed' in mmd_kwargs and not isinstance(mmd_kwargs['seed'], int):
            raise ValueError("seed must be an integer")
        
        self._mmd_kwargs = mmd_kwargs
        
            
    def compute(self, max_vectorize: int = 100) -> None:
        """
        Function that computes the MMD  metric (and its uncertainty) from two multivariate samples
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
    
    def Test_np(self, **mmd_kwargs) -> None:
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
            conditional_print(self.verbose, "Starting MMD metric calculation...")
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
        
        conditional_print(self.verbose, "Running numpy MMD calculation...")
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
            metric, metric_error = JMetrics.kpd(dist_1_k, dist_2_k, **mmd_kwargs)
            metric_list.append(metric)
            metric_error_list.append(metric_error)
            update_progress_bar()
        
        close_progress_bar()
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "MMD Test_np"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "numpy"}}
        result_value: Dict[str, Optional[DataTypeNP]] = {"metric_list": np.array(metric_list),
                                                         "metric_error_list": np.array(metric_error_list)}
        result: TwoSampleTestResult = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
        self.Results.append(result)
        
    def Test_tf(self, max_vectorize: int = 100) -> None:
        """
        Function that computes the MMD  metric (and its uncertainty) from two multivariate samples
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
            Given a value of max_vectorize, the niter MMD calculations are split in chunks of max_vectorize.
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
            conditional_tf_print(self.verbose, "Starting MMD metric calculation...")
            conditional_tf_print(self.verbose, "Running TF MMD calculation...")
            conditional_tf_print(self.verbose, "niter =", niter)
            conditional_tf_print(self.verbose, "batch_size =", batch_size)
            self._start = timer()

        def end_calculation() -> None:
            self._end = timer()
            elapsed = self.end - self.start
            conditional_tf_print(self.verbose, "MMD metric calculation completed in", str(elapsed), "seconds.")
                    
        def set_dist_num_from_symb(dist: DistTypeTF,
                                   nsamples: int,
                                   seed_generator: tf.random.Generator
                                  ) -> tf.Tensor:
            dist_num: tf.Tensor = generate_and_clean_data(dist, nsamples, self.Inputs.batch_size_gen, dtype = self.Inputs.dtype, seed_generator = seed_generator, strategy = self.Inputs.strategy) # type: ignore
            return dist_num
        
        def return_dist_num(dist_num: tf.Tensor) -> tf.Tensor:
            return dist_num

        @tf.function#(jit_compile=True, reduce_retracing = True)
        def batched_test_sub(dist_1_k_replica: tf.Tensor, 
                             dist_2_k_replica: tf.Tensor
                            ) -> DataTypeTF:
            def loop_body(idx):
                # Operations to be performed on each GPU
                vals = mmd_tf(dist_1_k_replica[idx, :, :], dist_2_k_replica[idx, :, :], **self.mmd_kwargs) # type: ignore
                vals = tf.cast(vals, dtype=dtype)
                return vals

            # Using tf.vectorized_map to parallelize operations across the first dimension of the input tensor
            vals_list: DataTypeTF = tf.vectorized_map(loop_body, tf.range(tf.shape(dist_1_k_replica)[0])) # type: ignore
            return vals_list
        
        @tf.function#(jit_compile=True, reduce_retracing = True)
        def batched_test(start: tf.Tensor, 
                         end: tf.Tensor,
                         seed_generator: tf.random.Generator
                        ) -> DataTypeTF:
            # Define batched distributions
            dist_1_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_1_symb, nsamples = batch_size * (end - start), seed_generator = seed_generator), # type: ignore
                                               false_fn = lambda: return_dist_num(dist_1_num[start * batch_size : end * batch_size, :])) # type: ignore
            dist_2_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_2_symb, nsamples = batch_size * (end - start), seed_generator = seed_generator), # type: ignore
                                               false_fn = lambda: return_dist_num(dist_2_num[start * batch_size : end * batch_size, :])) # type: ignore

            dist_1_k = tf.reshape(dist_1_k, (end - start, batch_size, ndims)) # type: ignore
            dist_2_k = tf.reshape(dist_2_k, (end - start, batch_size, ndims)) # type: ignore

            #if self.Inputs.strategy:
            #    per_replica_results = self.Inputs.strategy.run(batched_test_sub, args=(dist_1_k, dist_2_k))
            #    vals_list: tf.Tensor = self.Inputs.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_results, axis=None)
            #else:
            #vals_list = batched_test_sub(dist_1_k, dist_2_k)
            
            vals_list: DataTypeTF = batched_test_sub(dist_1_k, dist_2_k) # type: ignore
    
            return vals_list
        
        def compute_test(seed_generator: tf.random.Generator,
                         max_vectorize: int = 100
                        ) -> Tuple[DataTypeTF]:
            # Check if numerical distributions are empty and print a warning if so
            conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_1_num tensor is empty. Batches will be generated 'on-the-fly' from dist_1_symb.") # type: ignore
            conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_2_num tensor is empty. Batches will be generated 'on-the-fly' from dist_2_symb.") # type: ignore
            
            # Ensure that max_vectorize is an integer larger than ndims
            max_vectorize = tf.minimum(tf.cast(max_vectorize,tf.int32), tf.cast(niter,tf.int32)) # type: ignore

            # Compute the maximum number of iterations per chunk
            max_iter_per_chunk: tf.Tensor = max_vectorize # type: ignore
            
            # Compute the number of chunks
            nchunks: tf.Tensor = tf.cast(tf.math.ceil(tf.cast(niter, tf.int32) / max_iter_per_chunk), tf.int32) # type: ignore
            conditional_tf_print(tf.logical_and(self.verbose,tf.logical_not(tf.equal(nchunks,1))), "nchunks =", nchunks) # type: ignore

            res: tf.TensorArray = tf.TensorArray(dtype, size = nchunks)
            #res_vals: tf.TensorArray = tf.TensorArray(dtype, size = nchunks)

            #@tf.function
            def body(i: int, 
                     res: tf.TensorArray
                    ) -> Tuple[int, tf.TensorArray]:
                start: tf.Tensor = tf.cast(i * max_iter_per_chunk, tf.int32) # type: ignore
                end: tf.Tensor = tf.cast(tf.minimum(start + max_iter_per_chunk, niter), tf.int32) # type: ignore
                conditional_tf_print(tf.logical_and(tf.logical_or(tf.math.logical_not(tf.equal(start,0)),tf.math.logical_not(tf.equal(end,niter))), self.verbose), "Iterating from", start, "to", end, "out of", niter, ".") # type: ignore
                if self.Inputs.strategy:
                    per_replica_chunk_result = self.Inputs.strategy.run(batched_test, args=(start, end, seed_generator))
                    chunk_result: DataTypeTF = self.Inputs.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_chunk_result, axis=None)
                else:
                    chunk_result: DataTypeTF = batched_test(start, end, seed_generator = seed_generator) # type: ignore
                res = res.write(i, chunk_result)
                return i+1, res
            
            def cond(i: int, 
                     res: tf.TensorArray):
                return i < nchunks
            
            _, res = tf.while_loop(cond, body, [0, res])
            
            res_stacked: DataTypeTF = tf.reshape(res.stack(), (niter,-1))

            return res_stacked

        start_calculation()
        
        self.Inputs.reset_seed_generator()
        
        if self.Inputs.strategy:
            with self.Inputs.strategy.scope():
                seed_generator = tf.random.Generator.from_seed(self.Inputs.seed)
                vals_list: DataType = compute_test(seed_generator = seed_generator,
                                                   max_vectorize = max_vectorize)
        else:
            seed_generator = self.Inputs.seed_generator
            vals_list: DataType = compute_test(seed_generator = seed_generator,
                                               max_vectorize = max_vectorize)
                
        #print(f"vals_list: {vals_list=}")
        
        metric_list: DataTypeNP
        metric_error_list: DataTypeNP
        metric_list, metric_error_list = mmd_tf_output(vals_list)
                             
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "MMD Test_tf"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "tensorflow"}}
        result_value: Dict[str, Optional[DataTypeNP]] = {"metric_list": metric_list,
                                                         "metric_error_list": metric_error_list}
        result: TwoSampleTestResult = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
        self.Results.append(result)