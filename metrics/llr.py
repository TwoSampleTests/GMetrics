__all__ = ["LRMetric"]

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import traceback
from datetime import datetime
from timeit import default_timer as timer
from tqdm import tqdm # type: ignore
from .utils import reset_random_seeds
from .utils import conditional_print
from .utils import conditional_tf_print
from .utils import generate_and_clean_data
from .utils import NumpyDistribution
from .base import TwoSampleTestInputs
from .base import TwoSampleTestBase
from .base import TwoSampleTestResult
from .base import TwoSampleTestResults

from typing import Tuple, Union, Optional, Type, Dict, Any, List
from numpy import typing as npt
from .utils import DTypeType, IntTensor, FloatTensor, BoolTypeTF, BoolTypeNP, IntType, DataTypeTF, DataTypeNP, DataType, DistTypeTF, DistTypeNP, DistType, DataDistTypeNP, DataDistTypeTF, DataDistType, BoolType


def lr_statistic_np(logprob_ref_ref: DataTypeNP,
                    logprob_ref_alt: DataTypeNP,
                    logprob_alt_alt: DataTypeNP,
                   ) -> Tuple[np.float_, np.float_, np.float_, np.float_, np.float_]:
    # Count the number of samples
    n_ref: int = len(logprob_ref_ref)   
    n_alt: int = len(logprob_ref_alt)
    
    # Compute log likelihoods
    logprob_ref_ref_sum: np.float_ = np.sum(logprob_ref_ref)
    logprob_ref_alt_sum: np.float_ = np.sum(logprob_ref_alt)
    logprob_alt_alt_sum: np.float_ = np.sum(logprob_alt_alt)
    lik_ref_dist: np.float_ = logprob_ref_ref_sum + logprob_ref_alt_sum
    lik_alt_dist: np.float_ = logprob_ref_ref_sum + logprob_alt_alt_sum
    
    # Compute likelihood ratio statistic
    lik_ratio: np.float_ = 2 * (lik_alt_dist - lik_ref_dist)

    # Compute normalized likelihood ratio statistic
    n: float = 2 * n_ref * n_alt / (n_ref + n_alt)
    
    # Compute normalized likelihood ratio statistic
    lik_ratio_norm: np.float_ = lik_ratio / np.sqrt(n)
    
    return logprob_ref_ref_sum, logprob_ref_alt_sum, logprob_alt_alt_sum, lik_ratio, lik_ratio_norm

# If need to compile, should remove tf.print 'warning' statements. Doing some benchmarkings don't seem to improve performance.
# Left uncompiled for now.
@tf.function(jit_compile=True, reduce_retracing = True)
#@tf.function(reduce_retracing = True)
def lr_statistic_tf(logprob_ref_ref: DataTypeTF,
                    logprob_ref_alt: DataTypeTF,
                    logprob_alt_alt: DataTypeTF
                   ) -> Tuple[DataTypeTF, DataTypeTF, DataTypeTF, DataTypeTF, DataTypeTF]:
    # Count the number of samples
    n_ref: tf.Tensor = tf.cast(len(logprob_ref_ref), dtype = tf.int32) # type: ignore
    n_alt: tf.Tensor = tf.cast(len(logprob_ref_alt), dtype = tf.int32) # type: ignore
    
    # Compute log likelihoods
    logprob_ref_ref_sum: DataTypeTF = tf.reduce_sum(logprob_ref_ref)
    logprob_ref_alt_sum: DataTypeTF = tf.reduce_sum(logprob_ref_alt)
    logprob_alt_alt_sum: DataTypeTF = tf.reduce_sum(logprob_alt_alt)
    lik_ref_dist: DataTypeTF = logprob_ref_ref_sum + logprob_ref_alt_sum # type: ignore
    lik_alt_dist: DataTypeTF = logprob_ref_ref_sum + logprob_alt_alt_sum # type: ignore
    
    # Compute likelihood ratio statistic
    lik_ratio: DataTypeTF = 2 * (lik_alt_dist - lik_ref_dist) # type: ignore
    
    # Casting to float32 before performing division
    n_ref_float: tf.Tensor = tf.cast(n_ref, dtype = tf.float32) # type: ignore
    n_alt_float: tf.Tensor = tf.cast(n_alt, dtype = tf.float32) # type: ignore

    # Compute normalized likelihood ratio statistic
    n: tf.Tensor = 2 * n_ref_float * n_alt_float / (n_ref_float + n_alt_float) # type: ignore
    
    # Compute normalized likelihood ratio statistic
    lik_ratio_norm: DataTypeTF = lik_ratio / tf.sqrt(tf.cast(n, lik_ratio.dtype)) # type: ignore
    
    return logprob_ref_ref_sum, logprob_ref_alt_sum, logprob_alt_alt_sum, lik_ratio, lik_ratio_norm


class LRMetric(TwoSampleTestBase):
    """
    Class for computing the Likelihood Ratio (LR) metric.
    The metric can be computed only if the `is_symb_1` and `is_symb_2` attributes of the `data_input` object are True.
    In the opposite case the metric is not computed and the results are set to None.
    It inherits from the TwoSampleTestBase class.
    The LR is computed by first computing the log probabilities of the reference and alternative samples
    under the reference and alternative distributions, and then computing the LR statistic.
    The LR statistic can be computed using either numpy or tensorflow.
    
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
    compute() -> None
        Function that computes the LR metric selecting among the Test_np and Test_tf methods depending on the value of the use_tf attribute.
        
    Test_np() -> None
        Function that computes the LR metric using numpy functions.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.
        
    Test_tf() -> None
        Function that computes the LR metric using tensorflow functions.
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
        data_input = GMetrics.TwoSampleTestInputs(dist_1_input = dist_1_symb,
                                                  dist_2_input = dist_2_symb,
                                                  niter = 100,
                                                  batch_size = 10_000,
                                                  dtype_input = tf.float64,
                                                  seed_input = 0,
                                                  use_tf = True,
                                                  verbose = True)

        # Compute LR metric
        LR_metric = GMetrics.LRMetric(data_input = data_input, 
                                      progress_bar = True, 
                                      verbose = True)
        LR_metric.compute()
        LR_metric.Results[0].result_value
        
        >> {"logprob_ref_ref_sum_list": ...,
            "logprob_ref_alt_sum_list": ..., 
            "logprob_alt_alt_sum_list": ...,
            "lik_ratio_list": ...,
            "lik_ratio_norm_list": ...}
    """
    def __init__(self, 
                 data_input: TwoSampleTestInputs,
                 null_test: bool = False,
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
        # New attributes
        self._null: bool = null_test
    
        super().__init__(data_input = data_input, 
                         progress_bar = progress_bar,
                         verbose = verbose)
        
        if self.null:
            print("To run under the alternative hypothesis, set the `null_test` attribute to False.")
        else:
            print("To run under the null hypothesis, set the `null_test` attribute to True.")
        
    @property
    def null(self) -> bool:
        return self._null
        
    def compute(self) -> None:
        """
        Function that computes the LR metric selecting among the Test_np and Test_tf 
        methods depending on the value of the use_tf attribute.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

        """
        if self.use_tf:
            self.Test_tf()
        else:
            self.Test_np()
    
    def Test_np(self) -> None:
        """
        Function that computes the LR metric using numpy functions.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        # Utility function
        def stop_calculation() -> None:
            timestamp: str = datetime.now().isoformat()
            test_name: str = "LR Test_np"
            parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "numpy", "note": "test skipped because the inputs are not symbolic."}}
            result_value: Dict[str, Optional[DataTypeTF]] = {"logprob_ref_ref_sum_list": None,
                                                             "logprob_ref_alt_sum_list": None, 
                                                             "logprob_alt_alt_sum_list": None,
                                                             "lik_ratio_list": None,
                                                             "lik_ratio_norm_list": None} # type: ignore
            result: TwoSampleTestResult = TwoSampleTestResult(timestamp, test_name, parameters, result_value) # type: ignore
            self.Results.append(result)
            raise TypeError("LR metric can be computed only if the inputs are symbolic tfd.Distribution objects. Metric result has been set to `None`.")
        
        # Set alias for inputs
        if isinstance(self.Inputs.dist_1_symb, tfp.distributions.Distribution):
            dist_1_symb: tfp.distributions.Distribution = self.Inputs.dist_1_symb
        else:
            stop_calculation()
        if isinstance(self.Inputs.dist_2_symb, tfp.distributions.Distribution):
            dist_2_symb: tfp.distributions.Distribution = self.Inputs.dist_2_symb
        else:
            stop_calculation()
        if not self.Inputs.is_symb_1 or not self.Inputs.is_symb_2:
            stop_calculation()
        ndims: int = self.Inputs.ndims
        null: bool = self.null
        niter: int
        batch_size: int
        niter, batch_size = self.get_niter_batch_size_np() # type: ignore
        if isinstance(self.Inputs.dtype, tf.DType):
            dtype: Union[type, np.dtype] = self.Inputs.dtype.as_numpy_dtype
        else:
            dtype = self.Inputs.dtype
        dist_1_k: DataTypeTF
        dist_2_k: DataTypeTF
        logprob_ref_ref_filtered: DataTypeNP
        logprob_ref_alt_filtered: DataTypeNP
        logprob_alt_alt_filtered: DataTypeNP
        logprob_ref_ref_filtered_extra: DataTypeNP
        logprob_ref_alt_filtered_extra: DataTypeNP
        logprob_alt_alt_filtered_extra: DataTypeNP
        logprob_ref_ref_sum: np.float_
        logprob_ref_alt_sum: np.float_
        logprob_alt_alt_sum: np.float_
        lik_ratio: np.float_
        lik_ratio_norm: np.float_
        
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
            conditional_print(self.verbose, "Starting LR metric calculation...")
            conditional_print(self.verbose, "niter = {}" .format(niter))
            conditional_print(self.verbose, "batch_size = {}" .format(batch_size))
            self._start = timer()
            
        def init_progress_bar() -> None:
            nonlocal niter
            if self.progress_bar:
                self.pbar = tqdm(total = niter, desc = "Iterations")

        def update_progress_bar() -> None:
            if not self.pbar.disable:
                self.pbar.update(1)

        def close_progress_bar() -> None:
            if not self.pbar.disable:
                self.pbar.close()

        def end_calculation() -> None:
            self._end = timer()
            conditional_print(self.verbose, "Two-sample test calculation completed in "+str(self.end-self.start)+" seconds.")
                
        logprob_ref_ref_sum_list: List[np.float_] = []
        logprob_ref_alt_sum_list: List[np.float_] = []
        logprob_alt_alt_sum_list: List[np.float_] = []
        lik_ratio_list: List[np.float_] = []
        lik_ratio_norm_list: List[np.float_] = []

        start_calculation()
        init_progress_bar()
            
        self.Inputs.reset_seed_generator()
        
        conditional_print(self.verbose, "Running numpy LR calculation...")
        
        def get_logprobs(dist_1_num: DataType,
                         dist_2_num: DataType,
                         dist_1_symb: tfp.distributions.Distribution,
                         dist_2_symb: tfp.distributions.Distribution
                        ) -> Tuple[DataTypeNP, DataTypeNP, DataTypeNP]:
            # Convert to numpy arrays
            dist_1_num_np: DataTypeNP = np.array(dist_1_num) # type: ignore
            dist_2_num_np: DataTypeNP = np.array(dist_2_num) # type: ignore
            
            # Compute log probabilities
            logprob_ref_ref: DataTypeNP = dist_1_symb.log_prob(dist_1_num_np).numpy() # type: ignore
            logprob_ref_alt: DataTypeNP = dist_1_symb.log_prob(dist_2_num_np).numpy() # type: ignore
            logprob_alt_alt: DataTypeNP = dist_2_symb.log_prob(dist_2_num_np).numpy() # type: ignore
            
            # Create masks for finite log probabilities
            finite_indices_ref_ref: npt.NDArray[np.bool_] = np.isfinite(logprob_ref_ref)
            finite_indices_ref_alt: npt.NDArray[np.bool_] = np.isfinite(logprob_ref_alt)
            finite_indices_alt_alt: npt.NDArray[np.bool_] = np.isfinite(logprob_alt_alt)
            combined_finite_indices: npt.NDArray[np.bool_] = finite_indices_ref_ref & finite_indices_ref_alt & finite_indices_alt_alt
                        
            # Extract finite log probabilities
            logprob_ref_ref_filtered: DataTypeNP = logprob_ref_ref[combined_finite_indices]
            logprob_ref_alt_filtered: DataTypeNP = logprob_ref_alt[combined_finite_indices]
            logprob_alt_alt_filtered: DataTypeNP = logprob_alt_alt[combined_finite_indices]
            
            return logprob_ref_ref_filtered, logprob_ref_alt_filtered, logprob_alt_alt_filtered
        
        for k in range(niter):
            dist_1_k = tf.cast(set_dist_num_from_symb(dist = dist_1_symb, nsamples = batch_size, dtype = dtype), dtype = dtype) # type: ignore
            if null:
                dist_2_k = tf.cast(set_dist_num_from_symb(dist = dist_1_symb, nsamples = batch_size, dtype = dtype), dtype = dtype) # type: ignore
            else:
                dist_2_k = tf.cast(set_dist_num_from_symb(dist = dist_2_symb, nsamples = batch_size, dtype = dtype), dtype = dtype)
                
            # Compute log probabilities
            logprob_ref_ref_filtered, logprob_ref_alt_filtered, logprob_alt_alt_filtered = get_logprobs(dist_1_num = dist_1_k,
                                                                                                        dist_2_num = dist_2_k,
                                                                                                        dist_1_symb = dist_1_symb, # type: ignore
                                                                                                        dist_2_symb = dist_2_symb) # type: ignore
            
            # Count the number of finite samples
            n_finite: int = len(logprob_ref_ref_filtered)
            
            max_iter: int = 100
            iter: int = 0
            
            while n_finite < batch_size and iter < max_iter:
                iter += 1
                num_missing: int = batch_size - n_finite
                fraction: float = num_missing / batch_size
                print(f"Warning: Removed a fraction {fraction} of samples due to non-finite log probabilities.")
                
                # Generate extra samples
                dist_1_k_extra: tf.Tensor = tf.cast(dist_1_symb.sample(num_missing), dtype=dtype) # type: ignore
                if null:
                    dist_2_k_extra: tf.Tensor = tf.cast(dist_1_symb.sample(num_missing), dtype=dtype) # type: ignore
                else:
                    dist_2_k_extra: tf.Tensor = tf.cast(dist_2_symb.sample(num_missing), dtype=dtype) # type: ignore
                
                # Compute log probabilities
                logprob_ref_ref_filtered_extra, logprob_ref_alt_filtered_extra, logprob_alt_alt_filtered_extra = get_logprobs(dist_1_num = dist_1_k_extra,
                                                                                                                              dist_2_num = dist_2_k_extra,
                                                                                                                              dist_1_symb = dist_1_symb, # type: ignore
                                                                                                                              dist_2_symb = dist_2_symb) # type: ignore
                                
                # Append extra samples to the filtered log probabilities
                n_finite_extra: int = len(logprob_ref_ref_filtered_extra)
                if n_finite_extra > 0:
                    logprob_ref_ref_filtered = np.concatenate([logprob_ref_ref_filtered, logprob_ref_ref_filtered_extra])
                    logprob_ref_alt_filtered = np.concatenate([logprob_ref_alt_filtered, logprob_ref_alt_filtered_extra])
                    logprob_alt_alt_filtered = np.concatenate([logprob_alt_alt_filtered, logprob_alt_alt_filtered_extra])
                    
                # Count the number of finite samples
                n_finite = len(logprob_ref_ref_filtered)
            
            logprob_ref_ref_sum, logprob_ref_alt_sum, logprob_alt_alt_sum, lik_ratio, lik_ratio_norm = lr_statistic_np(logprob_ref_ref_filtered,
                                                                                                                       logprob_ref_alt_filtered, 
                                                                                                                       logprob_alt_alt_filtered) # type: ignore
            logprob_ref_ref_sum_list.append(logprob_ref_ref_sum)
            logprob_ref_alt_sum_list.append(logprob_ref_alt_sum)
            logprob_alt_alt_sum_list.append(logprob_alt_alt_sum)
            lik_ratio_list.append(lik_ratio)
            lik_ratio_norm_list.append(lik_ratio_norm)
            update_progress_bar()
        
        close_progress_bar()
        end_calculation()
        
        timestamp = datetime.now().isoformat()
        test_name = "LR Test_np"
        if self.null:
            test_type_dict: Dict[str,str] = {"test_type": "null test"}
        else:
            test_type_dict = {"test_type": "alternative test"}
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "numpy"}, **test_type_dict}
        result_value = {"logprob_ref_ref_sum_list": np.array(logprob_ref_ref_sum_list),
                        "logprob_ref_alt_sum_list": np.array(logprob_ref_alt_sum_list),
                        "logprob_alt_alt_sum_list": np.array(logprob_alt_alt_sum_list),
                        "lik_ratio_list": np.array(lik_ratio_list),
                        "lik_ratio_norm_list": np.array(lik_ratio_norm_list)} # type: ignore
        result: TwoSampleTestResult = TwoSampleTestResult(timestamp, test_name, parameters, result_value) # type: ignore
        self.Results.append(result)
        
    def Test_tf(self, max_vectorize: int = 100) -> None:
        """
        Function that computes the Frobenuis norm between the correlation matrices 
        of the two samples using tensorflow functions.
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
        # Utility function
        def stop_calculation() -> None:
            timestamp: str = datetime.now().isoformat()
            test_name: str = "LR Test_np"
            parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "numpy", "note": "test skipped because the inputs are not symbolic."}}
            result_value: Dict[str, Optional[DataTypeTF]] = {"logprob_ref_ref_sum_list": None,
                                                             "logprob_ref_alt_sum_list": None, 
                                                             "logprob_alt_alt_sum_list": None,
                                                             "lik_ratio_list": None,
                                                             "lik_ratio_norm_list": None} # type: ignore
            result: TwoSampleTestResult = TwoSampleTestResult(timestamp, test_name, parameters, result_value) # type: ignore
            self.Results.append(result)
            raise TypeError("LR metric can be computed only if the inputs are symbolic tfd.Distribution objects. Metric result has been set to `None`.")
        
        # Set alias for inputs
        if isinstance(self.Inputs.dist_1_symb, tfp.distributions.Distribution):
            dist_1_symb: tfp.distributions.Distribution = self.Inputs.dist_1_symb
        else:
            stop_calculation()
        if isinstance(self.Inputs.dist_2_symb, tfp.distributions.Distribution):
            dist_2_symb: tfp.distributions.Distribution = self.Inputs.dist_2_symb
        else:
            stop_calculation()
        if not self.Inputs.is_symb_1 or not self.Inputs.is_symb_2:
            stop_calculation()
        ndims: int = self.Inputs.ndims
        null: bool = self.null
        niter: int
        batch_size: int
        niter, batch_size = [int(i) for i in self.get_niter_batch_size_tf()] # type: ignore
        dtype: tf.DType = tf.as_dtype(self.Inputs.dtype)
        
        # Utility functions
        def start_calculation() -> None:
            conditional_tf_print(self.verbose, "\n------------------------------------------")
            conditional_tf_print(self.verbose, "Starting LR metric calculation...")
            conditional_tf_print(self.verbose, "Running TF LR calculation...")
            conditional_tf_print(self.verbose, "niter =", niter)
            conditional_tf_print(self.verbose, "batch_size =", batch_size)
            self._start = timer()

        def end_calculation() -> None:
            self._end = timer()
            elapsed = self.end - self.start
            conditional_tf_print(self.verbose, "LR metric calculation completed in", str(elapsed), "seconds.")
            
        def set_dist_num_from_symb(dist: DistTypeTF,
                                   nsamples: int,
                                  ) -> tf.Tensor:
            dist_num: tf.Tensor = generate_and_clean_data(dist, nsamples, self.Inputs.batch_size_gen, dtype = self.Inputs.dtype, seed_generator = self.Inputs.seed_generator, strategy = self.Inputs.strategy) # type: ignore
            return dist_num
        
        #@tf.function(jit_compile=True, reduce_retracing=True)
        @tf.function
        def get_logprobs(dist_1_num: DataTypeTF,
                         dist_2_num: DataTypeTF,
                         dist_1_symb: tfp.distributions.Distribution,
                         dist_2_symb: tfp.distributions.Distribution
                        ) -> Tuple[DataTypeTF, DataTypeTF, DataTypeTF]:
            # Compute log probabilities
            logprob_ref_ref: DataTypeTF = dist_1_symb.log_prob(dist_1_num)
            logprob_ref_alt: DataTypeTF = dist_1_symb.log_prob(dist_2_num)
            logprob_alt_alt: DataTypeTF = dist_2_symb.log_prob(dist_2_num)
            
            # Create masks for finite log probabilities
            finite_indices_ref_ref: tf.Tensor = tf.math.is_finite(logprob_ref_ref) # type: ignore
            finite_indices_ref_alt: tf.Tensor = tf.math.is_finite(logprob_ref_alt) # type: ignore
            finite_indices_alt_alt: tf.Tensor = tf.math.is_finite(logprob_alt_alt) # type: ignore
            combined_finite_indices: tf.Tensor = tf.logical_and(tf.logical_and(finite_indices_ref_ref, finite_indices_ref_alt), finite_indices_alt_alt) # type: ignore
                        
            # Extract finite log probabilities
            logprob_ref_ref_filtered: DataTypeTF = logprob_ref_ref[combined_finite_indices] # type: ignore
            logprob_ref_alt_filtered: DataTypeTF = logprob_ref_alt[combined_finite_indices] # type: ignore
            logprob_alt_alt_filtered: DataTypeTF = logprob_alt_alt[combined_finite_indices] # type: ignore
            
            return logprob_ref_ref_filtered, logprob_ref_alt_filtered, logprob_alt_alt_filtered
        
        @tf.function
        def batched_test_sub(logprob_ref_ref: tf.Tensor, 
                             logprob_ref_alt: tf.Tensor,
                             logprob_alt_alt: tf.Tensor
                            ) -> DataTypeTF:
            def loop_body(idx):
                logprob_ref_ref_sum, logprob_ref_alt_sum, logprob_alt_alt_sum, lik_ratio, lik_ratio_norm = lr_statistic_tf(logprob_ref_ref[idx, :], # type: ignore
                                                                                                                           logprob_ref_alt[idx, :], # type: ignore
                                                                                                                           logprob_alt_alt[idx, :]) # type: ignore
                logprob_ref_ref_sum = tf.cast(logprob_ref_ref_sum, dtype = dtype)
                logprob_ref_alt_sum = tf.cast(logprob_ref_alt_sum, dtype = dtype)
                logprob_alt_alt_sum = tf.cast(logprob_alt_alt_sum, dtype = dtype)
                lik_ratio = tf.cast(lik_ratio, dtype = dtype)
                lik_ratio_norm = tf.cast(lik_ratio_norm, dtype = dtype)
                return logprob_ref_ref_sum, logprob_ref_alt_sum, logprob_alt_alt_sum, lik_ratio, lik_ratio_norm

            # Vectorize over ndims*chunk_size
            logprob_ref_ref_sum, logprob_ref_alt_sum, logprob_alt_alt_sum, lik_ratio, lik_ratio_norm = tf.vectorized_map(loop_body, tf.range(tf.shape(logprob_ref_ref)[0])) # type: ignore

            logprob_ref_ref_sum = tf.expand_dims(logprob_ref_ref_sum, axis=1)
            logprob_ref_alt_sum = tf.expand_dims(logprob_ref_alt_sum, axis=1)
            logprob_alt_alt_sum = tf.expand_dims(logprob_alt_alt_sum, axis=1)
            lik_ratio = tf.expand_dims(lik_ratio, axis=1)
            lik_ratio_norm = tf.expand_dims(lik_ratio_norm, axis=1)
            
            res: DataTypeTF = tf.concat([logprob_ref_ref_sum, logprob_ref_alt_sum, logprob_alt_alt_sum, lik_ratio, lik_ratio_norm], axis=1) # type: ignore
            return res
        
        def batched_test(start: tf.Tensor, 
                         end: tf.Tensor
                        ) -> DataTypeTF:
            # Define the loop body to vectorize over ndims*chunk_size
            dist_1_k: DataTypeTF = set_dist_num_from_symb(dist_1_symb, nsamples = batch_size * (end - start)) # type: ignore
            if null:
                dist_2_k: DataTypeTF = set_dist_num_from_symb(dist_1_symb, nsamples = batch_size * (end - start)) # type: ignore
            else:
                dist_2_k: DataTypeTF = set_dist_num_from_symb(dist_2_symb, nsamples = batch_size * (end - start)) # type: ignore
            
            # Compute log probabilities
            logprob_ref_ref_filtered: DataTypeTF
            logprob_ref_alt_filtered: DataTypeTF
            logprob_alt_alt_filtered: DataTypeTF
            logprob_ref_ref_filtered, logprob_ref_alt_filtered, logprob_alt_alt_filtered = get_logprobs(dist_1_num = dist_1_k, 
                                                                                                        dist_2_num = dist_2_k, 
                                                                                                        dist_1_symb = dist_1_symb, # type: ignore
                                                                                                        dist_2_symb = dist_2_symb) # type: ignore
            
            # Count the number of finite samples
            n_finite: tf.Tensor = tf.shape(logprob_ref_ref_filtered)[0]
            
            max_iter: int = 100
            iter: int = 0
            
            while n_finite < batch_size and iter < max_iter:
                iter += 1
                num_missing: int = batch_size - n_finite
                fraction: float = num_missing / batch_size
                
                # Generate extra samples
                dist_1_k_extra: DataTypeTF = tf.cast(set_dist_num_from_symb(dist_1_symb, nsamples = num_missing), dtype=dtype) # type: ignore
                if null:
                    dist_2_k_extra: DataTypeTF = tf.cast(set_dist_num_from_symb(dist_1_symb, nsamples = num_missing), dtype=dtype) # type: ignore
                else:
                    dist_2_k_extra: DataTypeTF = tf.cast(set_dist_num_from_symb(dist_2_symb, nsamples = num_missing), dtype=dtype) # type: ignore
                
                # Compute log probabilities
                logprob_ref_ref_filtered_extra: DataTypeTF
                logprob_ref_alt_filtered_extra: DataTypeTF
                logprob_alt_alt_filtered_extra: DataTypeTF
                logprob_ref_ref_filtered_extra, logprob_ref_alt_filtered_extra, logprob_alt_alt_filtered_extra = get_logprobs(dist_1_num = dist_1_k_extra,
                                                                                                                              dist_2_num = dist_2_k_extra,
                                                                                                                              dist_1_symb = dist_1_symb, # type: ignore
                                                                                                                              dist_2_symb = dist_2_symb) # type: ignore
                                
                # Append extra samples to the filtered log probabilities
                n_finite_extra: tf.Tensor = tf.shape(logprob_ref_ref_filtered_extra)[0]
                if n_finite_extra > 0:
                    logprob_ref_ref_filtered = tf.concat([logprob_ref_ref_filtered, logprob_ref_ref_filtered_extra], axis = 0) # type: ignore
                    logprob_ref_alt_filtered = tf.concat([logprob_ref_alt_filtered, logprob_ref_alt_filtered_extra], axis = 0) # type: ignore
                    logprob_alt_alt_filtered = tf.concat([logprob_alt_alt_filtered, logprob_alt_alt_filtered_extra], axis = 0) # type: ignore
                    
                # Count the number of finite samples
                n_finite: tf.Tensor = tf.shape(logprob_ref_ref_filtered)[0]
                
            logprob_ref_ref_filtered_reshaped: tf.Tensor = tf.reshape(logprob_ref_ref_filtered, (end - start, batch_size)) # type: ignore
            logprob_ref_alt_filtered_reshaped: tf.Tensor = tf.reshape(logprob_ref_alt_filtered, (end - start, batch_size)) # type: ignore
            logprob_alt_alt_filtered_reshaped: tf.Tensor = tf.reshape(logprob_alt_alt_filtered, (end - start, batch_size)) # type: ignore

            res: DataTypeTF = batched_test_sub(logprob_ref_ref_filtered_reshaped, 
                                               logprob_ref_alt_filtered_reshaped, 
                                               logprob_alt_alt_filtered_reshaped) # type: ignore
            
            return res
        
        #@tf.function(reduce_retracing=True)
        def compute_test(max_vectorize: int = 100) -> Tuple[DataTypeTF, DataTypeTF, DataTypeTF, DataTypeTF, DataTypeTF]:            
            # Ensure that max_vectorize is an integer larger than ndims
            max_vectorize = int(tf.cast(tf.minimum(max_vectorize, niter),tf.int32)) # type: ignore

            # Compute the maximum number of iterations per chunk
            max_iter_per_chunk: int = max_vectorize # type: ignore

            # Compute the number of chunks
            nchunks: int = int(tf.cast(tf.math.ceil(niter / max_iter_per_chunk), tf.int32)) # type: ignore
            conditional_tf_print(tf.logical_and(self.verbose,tf.logical_not(tf.equal(nchunks,1))), "nchunks =", nchunks) # type: ignore
            
            # Initialize the result TensorArray
            res: tf.TensorArray = tf.TensorArray(dtype, size = nchunks)
            logprob_ref_ref_sum = tf.TensorArray(dtype, size = nchunks)
            logprob_ref_alt_sum = tf.TensorArray(dtype, size = nchunks)
            logprob_alt_alt_sum = tf.TensorArray(dtype, size = nchunks)
            lik_ratio = tf.TensorArray(dtype, size = nchunks)
            lik_ratio_norm = tf.TensorArray(dtype, size = nchunks)
            
            def body(i, res):
                start = i * max_iter_per_chunk
                end = tf.minimum(start + max_iter_per_chunk, niter)
                conditional_tf_print(tf.logical_and(tf.logical_or(tf.math.logical_not(tf.equal(start,0)),tf.math.logical_not(tf.equal(end,niter))), self.verbose), "Iterating from", start, "to", end, "out of", niter, ".") # type: ignore
                chunk_result = batched_test(start, end) # type: ignore
                res = res.write(i, chunk_result)
                return i+1, res
    
            _, res = tf.while_loop(lambda i, _: i < nchunks, body, [0, res])
            
            for i in range(nchunks):
                res_i = res.read(i)
                logprob_ref_ref_sum = logprob_ref_ref_sum.write(i, res_i[:, 0])
                logprob_ref_alt_sum = logprob_ref_alt_sum.write(i, res_i[:, 1])
                logprob_alt_alt_sum = logprob_alt_alt_sum.write(i, res_i[:, 2])
                lik_ratio = lik_ratio.write(i, res_i[:, 3])
                lik_ratio_norm = lik_ratio_norm.write(i, res_i[:, 4])
                
            logprob_ref_ref_sum_stacked: DataTypeTF = tf.reshape(logprob_ref_ref_sum.stack(), [-1])
            logprob_ref_alt_sum_stacked: DataTypeTF = tf.reshape(logprob_ref_alt_sum.stack(), [-1])
            logprob_alt_alt_sum_stacked: DataTypeTF = tf.reshape(logprob_alt_alt_sum.stack(), [-1])
            lik_ratio_stacked: DataTypeTF = tf.reshape(lik_ratio.stack(), [-1])
            lik_ratio_norm_stacked: DataTypeTF = tf.reshape(lik_ratio_norm.stack(), [-1])
        
            return logprob_ref_ref_sum_stacked, logprob_ref_alt_sum_stacked, logprob_alt_alt_sum_stacked, lik_ratio_stacked, lik_ratio_norm_stacked

        start_calculation()
        
        self.Inputs.reset_seed_generator()
        
        logprob_ref_ref_sum_list: DataTypeTF
        logprob_ref_alt_sum_list: DataTypeTF
        logprob_alt_alt_sum_list: DataTypeTF
        lik_ratio_list: DataTypeTF
        lik_ratio_norm_list: DataTypeTF
        logprob_ref_ref_sum_list, logprob_ref_alt_sum_list, logprob_alt_alt_sum_list, lik_ratio_list, lik_ratio_norm_list = compute_test(max_vectorize = max_vectorize) # type: ignore
                             
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "LR Test_tf"
        if self.null:
            test_type_dict: Dict[str,str] = {"test_type": "null test"}
        else:
            test_type_dict = {"test_type": "alternative test"}
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "tensorflow"}, **test_type_dict}
        result_value: Dict[str, Optional[DataTypeNP]] = {"logprob_ref_ref_sum_list": logprob_ref_ref_sum_list.numpy(),
                                                          "logprob_ref_alt_sum_list": logprob_ref_alt_sum_list.numpy(),
                                                          "logprob_alt_alt_sum_list": logprob_alt_alt_sum_list.numpy(),
                                                          "lik_ratio_list": lik_ratio_list.numpy(),
                                                          "lik_ratio_norm_list": lik_ratio_norm_list.numpy()}
        result: TwoSampleTestResult = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
        self.Results.append(result)