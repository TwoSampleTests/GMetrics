__all__ = ["wasserstein_distance_tf",
           "swd_2samp_tf",
           "SWDMetric"]

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import traceback
from datetime import datetime
from timeit import default_timer as timer
from tqdm import tqdm # type: ignore
from scipy.stats import wasserstein_distance # type: ignore
from GMetrics.utils import reset_random_seeds
from GMetrics.utils import conditional_print
from GMetrics.utils import conditional_tf_print
from GMetrics.utils import generate_and_clean_data
from GMetrics.utils import NumpyDistribution
from GMetrics.base import TwoSampleTestInputs
from GMetrics.base import TwoSampleTestSlicedBase
from GMetrics.base import TwoSampleTestResult
from GMetrics.base import TwoSampleTestResults

from typing import Tuple, Union, Optional, Type, Dict, Any, List
from GMetrics.utils import DTypeType, IntTensor, FloatTensor, BoolTypeTF, BoolTypeNP, IntType, DataTypeTF, DataTypeNP, DataType, DistTypeTF, DistTypeNP, DistType, DataDistTypeNP, DataDistTypeTF, DataDistType, BoolType

@tf.function(jit_compile=True, reduce_retracing=True)
def wasserstein_distance_tf(data1: tf.Tensor, 
                            data2: tf.Tensor
                           ) -> tf.Tensor:
    """
    Function that computes the Wasserstein distance between two 1D samples.
    Result is the samme as scipy.stats.wasserstein_distance.
    
    The tf.function decorator is used to speed up subsequent calls to this function and to avoid retracing.
    
    Parameters:
    -----------
    data1: tf.Tensor, optional, shape=(n1,)
        First sample. Sample sizes can be different.
        
    data2: tf.Tensor, optional, shape=(n2,)
        Second sample. Sample sizes can be different.

    Returns:
    --------
    wd: tf.Tensor, shape=(1,)
        The Wasserstein distance (Earth Mover's Distance) between the samples.
    """
    data1 = tf.sort(data1)
    data2 = tf.sort(data2)
    # calculate the differences between corresponding points in the sorted distributions
    diff = tf.abs(data1 - data2)
    # calculate the mean of these differences
    wd = tf.reduce_mean(diff)    
    return wd

@tf.function(jit_compile=True, reduce_retracing = True)
def swd_2samp_tf(data1: tf.Tensor, 
                 data2: tf.Tensor,
                 directions_input: DataDistType
                ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Function that computes the sliced Wasserstein distance between two N-dimensional samples.
    The sliced Wasserstein distance is computed by projecting the samples 
    onto random directions, computing the Wasserstein distance between the projections, and
    then taking the mean and standard deviation of the Wasserstein distances.
    The sliced Wasserstein distances can be computed using either numpy or tensorflow.
    
    The tf.function decorator is used to speed up subsequent calls to this function and to avoid retracing.

    Parameters:
    -----------
    data1: tf.Tensor, optional, shape = (n1,)
        First sample. Sample sizes can be different.
        
    data2: tf.Tensor, optional, shape = (n2,)
        Second sample. Sample sizes can be different.

    nslices: int, optional, default = 100
        Number of random directions to use for the projection.

    Returns:
    --------
    result: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        Tuple containing the following elements:

        - swd_mean: tf.Tensor, shape = (1,)
            Mean of the Wasserstein distances between the projections.

        - swd_std: tf.Tensor, shape = (1,)
            Standard deviation of the Wasserstein distances between the projections.

        - swd_proj: tf.Tensor, shape = (nslices,)
            Wasserstein distances between the projections.
    """
    # Compute ndims
    ndims = tf.shape(data1)[1]
    
    # Cast random directions
    directions = tf.cast(directions_input, dtype=data1.dtype)
    
    # Compute projections for all directions at once
    data1_proj = tf.tensordot(data1, directions, axes=[[1],[1]])
    data2_proj = tf.tensordot(data2, directions, axes=[[1],[1]])
    
    # Transpose the projection tensor to have slices on the first axis
    data1_proj = tf.transpose(data1_proj)
    data2_proj = tf.transpose(data2_proj)
    
    # Apply wasserstein_distance to each slice using tf.vectorized_map
    swd_proj = tf.vectorized_map(lambda args: wasserstein_distance_tf(*args), (data1_proj, data2_proj)) # type: ignore
    
    # Compute mean and std
    swd_mean = tf.reduce_mean(swd_proj)
    swd_std = tf.math.reduce_std(swd_proj)

    return swd_mean, swd_std, swd_proj # type: ignore


class SWDMetric(TwoSampleTestSlicedBase):
    """
    Class for computing the Sliced Wasserstein Distance (SWD) between two samples.
    It inherits from the TwoSampleTestSlicedBase class.
    The SWD is computed by projecting the samples onto random directions, 
    computing the Wasserstein distance between the projections, and
    then taking the mean and standard deviation of the Wasserstein distances.
    The sliced Wasserstein distances can be computed using either numpy or tensorflow.
    The scipy implementation is used for the numpy backend.
    A custom tensorflow implementation is used for the tensorflow backend.
    The tensorflow implementation is faster than the scipy implementation, especially for large sample sizes,
    number of projections, and number of iterations.
    
    Parameters:
    ----------
    data_input: TwoSampleTestInputs
        Object containing the inputs for the two-sample test.

    progress_bar: bool, optional, default = False
        If True, display a progress bar. The progress bar is automatically disabled when running tensorflow functions.
        
    verbose: bool, optional, default = False
        If True, print additional information.

    Attributes:
    ----------
    Inputs: TwoSampleTestInputs object
        Object containing the inputs for the two-sample test.

    Results: TwoSampleTestResults object
        Object containing the results of the two-sample test.

    start: float
        Time when the two-sample test calculation started.

    end: float
        Time when the two-sample test calculation ended.

    pbar: tqdm
        Progress bar object.
        
    Methods:
    -------
    compute(nslices: int = 100) -> None
        Function that computes the sliced Wasserstein distance between the two samples
        selecting among the Test_np and Test_tf methods depending on the value of the use_tf attribute.
        
    Test_np(nslices: int = 100) -> None
        Function that computes the sliced Wasserstein distance between the two samples using numpy functions.
        The number of random directions used for the projection is given by nslices.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.
        
    Test_tf(nslices: int = 100) -> None
        Function that computes the sliced Wasserstein distance between the two samples using tensorflow functions.
        The number of random directions used for the projection is given by nslices.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.
        
    Examples:
    --------
    
    .. code-block:: python
    
        import numpy as np
        import tensorflow as tf
        import tensorflow_probability as tfp
        import GenerativeModelsMetrics as GMetrics

        # Set random seed
        seed = 0
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Define inputs
        nsamples = 1_000_000
        ndims = 2
        dtype = tf.float32
        ndims = 100
        eps = 0.1
        dist_1_symb = tfp.distributions.Normal(loc=np.full(ndims,0.), scale=np.full(ndims,1.))
        dist_2_symb = tfp.distributions.Normal(loc=np.random.uniform(-eps, eps, ndims), scale=np.random.uniform(1-eps, 1+eps, ndims))
        dist_1_num = tf.cast(dist_1_symb.sample(nsamples),tf.float32)
        dist_2_num = tf.cast(dist_2_symb.sample(nsamples),tf.float32)
        data_input = GMetrics.TwoSampleTestInputs(dist_1_input = dist_1_num,
                                                  dist_2_input = dist_2_num,
                                                  niter = 100,
                                                  batch_size = 10_000,
                                                  dtype_input = tf.float64,
                                                  seed_input = 0,
                                                  use_tf = True,
                                                  verbose = True)

        # Compute SWD metric
        swd_metric = GMetrics.SWDMetric(data_input = data_input, 
                                        progress_bar = True, 
                                        verbose = True)
        swd_metric.compute(max_vectorize = int(1e6))
        swd_metric.Results[0].result_value
        
        >> {'metric_lists': [[...]]
            'metric_means': [...],
            'metric_stds': [...]}
    """
    def __init__(self, 
                 data_input: TwoSampleTestInputs,
                 nslices: int = 100,
                 seed_slicing: Optional[int] = None,
                 progress_bar: bool = False,
                 verbose: bool = False
                ) -> None:
        # From base class
        self._Inputs: TwoSampleTestInputs
        self._progress_bar: bool
        self._verbose: bool
        self._start: float
        self._end: float
        self._pbar: tqdm
        self._Results: TwoSampleTestResults
        self._seed_slicing: int
        self._nslices: int
        self._directions: DataTypeNP
    
        super().__init__(data_input = data_input, 
                         nslices = nslices,
                         seed_slicing = seed_slicing,
                         progress_bar = progress_bar,
                         verbose = verbose)
        
    def compute(self, 
                max_vectorize: int = 100
               ) -> None:
        """
        Function that computes the sliced Wasserstein distance between the two samples
        selecting among the Test_np and Test_tf methods depending on the value of the use_tf attribute.

        Parameters:
        -----------
        nslices: int, optional, default = 100
            Number of random directions to use for the projection.

        Returns:
        --------
        None

        """
        if self.use_tf:
            self.Test_tf(max_vectorize = max_vectorize)
        else:
            self.Test_np()
    
    def Test_np(self) -> None:
        """
        Function that computes the sliced Wasserstein distance between the two samples using numpy functions.
        The number of random directions used for the projection is given by nslices.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.

        Parameters:
        -----------
        nslices: int, optional, default = 100
            Number of random directions to use for the projection.

        Returns:
        --------
        None
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
            conditional_print(self.verbose, "Starting SWD metric calculation...")
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
        
        metric_lists: List[List] = []
        metric_means: List[float] = []
        metric_stds: List[float] = []

        start_calculation()
        init_progress_bar()
            
        self.Inputs.reset_seed_generator()
        
        conditional_print(self.verbose, "Running numpy SKS calculation...")
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
            # Compute sliced Wasserstein distance
            list1 = []
            for direction in self.directions:
                dist_1_proj = dist_1_k @ direction
                dist_2_proj = dist_2_k @ direction
                list1.append(wasserstein_distance(dist_1_proj, dist_2_proj))
            metric_lists.append(list1)
            metric_means.append(np.mean(list1)) # type: ignore
            metric_stds.append(np.std(list1)) # type: ignore
            update_progress_bar()
        
        close_progress_bar()
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "SWD Test_np"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "numpy"}}
        result_value: Dict[str, Optional[DataTypeNP]] = {"metric_lists": np.array(metric_lists), 
                                                         "metric_means": np.array(metric_means),
                                                         "metric_stds": np.array(metric_stds)}
        result: TwoSampleTestResult = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
        self.Results.append(result)
        
    def Test_tf(self, max_vectorize: int = 100) -> None:
        """
        Function that computes the sliced Wasserstein distance between the two samples using tensorflow functions.
        The number of random directions used for the projection is given by nslices.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.
        
        Parameters:
        -----------
        nslices: int, optional, default = 100
            Number of random directions to use for the projection.

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
            conditional_tf_print(self.verbose, "Starting SWD metric calculation...")
            conditional_tf_print(self.verbose, "Running TF SWD calculation...")
            conditional_tf_print(self.verbose, "niter =", niter)
            conditional_tf_print(self.verbose, "batch_size =", batch_size)
            self._start = timer()

        def end_calculation() -> None:
            self._end = timer()
            elapsed = self.end - self.start
            conditional_tf_print(self.verbose, "SWD metric calculation completed in", str(elapsed), "seconds.")
            
        def set_dist_num_from_symb(dist: DistTypeTF,
                                   nsamples: int,
                                  ) -> tf.Tensor:
            dist_num: tf.Tensor = generate_and_clean_data(dist, nsamples, self.Inputs.batch_size_gen, dtype = self.Inputs.dtype, seed_generator = self.Inputs.seed_generator, strategy = self.Inputs.strategy) # type: ignore
            return dist_num
        
        def return_dist_num(dist_num: tf.Tensor) -> tf.Tensor:
            return dist_num
        
        @tf.function
        def batched_test_sub(dist_1_k_replica: tf.Tensor, 
                             dist_2_k_replica: tf.Tensor
                            ) -> DataTypeTF:
            def loop_body(idx):
                swd_mean, swd_std, swd_proj = swd_2samp_tf(dist_1_k_replica[idx, :, :], dist_2_k_replica[idx, :, :], directions_input = self.directions) # type: ignore
                swd_mean = tf.cast(swd_mean, dtype = dtype)
                swd_std = tf.cast(swd_std, dtype = dtype)
                swd_proj = tf.cast(swd_proj, dtype = dtype)
                return swd_mean, swd_std, swd_proj

            # Vectorize over ndims*chunk_size
            swd_mean, swd_std, swd_proj = tf.vectorized_map(loop_body, tf.range(tf.shape(dist_1_k_replica)[0])) # type: ignore

            swd_mean = tf.expand_dims(swd_mean, axis=1)
            swd_std = tf.expand_dims(swd_std, axis=1)
            
            res: DataTypeTF = tf.concat([swd_mean, swd_std, swd_proj], axis=1) # type: ignore

            return res
        
        #@tf.function(jit_compile=True, reduce_retracing=True)
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
            
        #@tf.function(reduce_retracing=True)
        def compute_test(max_vectorize: int = 100) -> Tuple[DataTypeTF, DataTypeTF, DataTypeTF]:
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

            # Initialize the result TensorArray
            res = tf.TensorArray(dtype, size = nchunks)
            res_swd_mean = tf.TensorArray(dtype, size = nchunks)
            res_swd_std = tf.TensorArray(dtype, size = nchunks)
            res_swd_proj = tf.TensorArray(dtype, size = nchunks)
            
            def body(i, res):
                start = i * max_iter_per_chunk
                end = tf.minimum(start + max_iter_per_chunk, niter)
                conditional_tf_print(tf.logical_and(tf.logical_or(tf.math.logical_not(tf.equal(start,0)),tf.math.logical_not(tf.equal(end,niter))), self.verbose), "Iterating from", start, "to", end, "out of", niter, ".") # type: ignore
                chunk_result = batched_test(start, end) # type: ignore
                res = res.write(i, chunk_result)
                return i+1, res
    
            def cond(i, res):
                return i < nchunks
            
            _, res = tf.while_loop(cond, body, [0, res])
            
            for i in range(nchunks):
                res_i = res.read(i)
                res_swd_mean = res_swd_mean.write(i, res_i[:,0])
                res_swd_std = res_swd_std.write(i, res_i[:,1])
                res_swd_proj = res_swd_proj.write(i, res_i[:,2:])
                
            swd_means: DataTypeTF = tf.reshape(res_swd_mean.stack(), (niter,))
            swd_stds: DataTypeTF = tf.reshape(res_swd_std.stack(), (niter,))
            swd_lists: DataTypeTF = tf.reshape(res_swd_proj.stack(), (niter, -1))
                            
            return swd_means, swd_stds, swd_lists
                
        start_calculation()
        
        self.Inputs.reset_seed_generator()
        
        swd_means: DataTypeTF
        swd_stds: DataTypeTF
        swd_lists: DataTypeTF
        swd_means, swd_stds, swd_lists = compute_test(max_vectorize = max_vectorize)
                             
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "SWD Test_tf"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "tensorflow"}}
        result_value: Dict[str, Optional[DataTypeNP]] = {"metric_lists": swd_lists.numpy(),
                                                         "metric_means": swd_means.numpy(),
                                                         "metric_stds": swd_stds.numpy()}
        result: TwoSampleTestResult = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
        self.Results.append(result)