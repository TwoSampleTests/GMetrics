__all__ = ["get_logflk_config",
           "candidate_sigma",
           "trainer",
           "compute_t",   
           "NPLMMetric"]

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
from GMetrics.utils import reset_random_seeds
from GMetrics.utils import conditional_print
from GMetrics.utils import conditional_tf_print
from GMetrics.utils import generate_and_clean_data
from GMetrics.utils import NumpyDistribution
from GMetrics.base import TwoSampleTestInputs
from GMetrics.base import TwoSampleTestBase
from GMetrics.base import TwoSampleTestResult
from GMetrics.base import TwoSampleTestResults


from typing import Tuple, Union, Optional, Type, Dict, Any, List, Set
from GMetrics.utils import DTypeType, IntTensor, FloatTensor, BoolTypeTF, BoolTypeNP, IntType, DataTypeTF, DataTypeNP, DataType, DistTypeTF, DistTypeNP, DistType, DataDistTypeNP, DataDistTypeTF, DataDistType, BoolType

import torch
import time

from falkon import LogisticFalkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.gsc_losses import WeightedCrossEntropyLoss

from scipy.spatial.distance import pdist

def get_logflk_config(M, flk_sigma, lam, weight, iter, seed=None, cpu=False):     #iter=[1000000]
    # it returns logfalkon parameters
    return {
            'kernel' : GaussianKernel(sigma=flk_sigma),
            'M' : M, #number of Nystrom centers,
            'penalty_list' : [lam], # list of regularization parameters,
            'iter_list' : [iter], #list of number of CG iterations,
            'options' : FalkonOptions(cg_tolerance=np.sqrt(1e-7), keops_active='no', use_cpu=cpu, debug = False),
            'seed' : seed, # (int or None), the model seed (used for Nystrom center selection) is manually set,
            'loss' : WeightedCrossEntropyLoss(kernel=GaussianKernel(sigma=flk_sigma), neg_weight=weight),
            }

def candidate_sigma(data, perc=90):
    # this function gives an estimate of the width of the gaussian kernel
    # use on a (small) sample of reference data (standardize first if necessary)
    pairw = pdist(data)
    return round(np.percentile(pairw,perc),1)

def trainer(X, Y, flk_config):
    # trainer for logfalkon model
    Xtorch=torch.from_numpy(X) #features 
    Ytorch=torch.from_numpy(Y) #labels
    model = LogisticFalkon(**flk_config)
    model.fit(Xtorch, Ytorch)
    return model.predict(Xtorch).numpy()

def compute_t(preds, Y, weight):
    # it returns extended log likelihood ratio from predictions
    diff = weight*np.sum(1 - np.exp(preds[Y==0]))
    return 2 * (diff + np.sum(preds[Y==1]))


# @tf.function(jit_compile = True, reduce_retracing = True)
# def _normalise_features_tf(data1_input: DataType, 
#                            data2_input: Optional[DataType] = None
#                           ) -> Union[DataTypeTF, Tuple[DataTypeTF, DataTypeTF]]:
#     data1: DataTypeTF = tf.convert_to_tensor(data1_input)
#     maxes: tf.Tensor = tf.reduce_max(tf.abs(data1), axis=0)
#     maxes = tf.where(tf.equal(maxes, 0), tf.ones_like(maxes), maxes)  # don't normalize in case of features which are just 0

#     if data2_input is not None:
#         data2: DataTypeTF = tf.convert_to_tensor(data2_input)
#         return data1 / maxes, data2 / maxes
#     else:
#         return data1 / maxes

# @tf.function(jit_compile=True)
# def generate_unique_indices(num_samples, batch_size, num_batches, seed = None):
#     reset_random_seeds(seed)
#     # Edge case: if num_samples equals batch_size, shuffle uniquely for each batch
#     if num_samples == batch_size:
#         # Create a large tensor that repeats the range [0, num_samples] num_batches times
#         indices = tf.tile(tf.range(num_samples, dtype=tf.int32)[tf.newaxis, :], [num_batches, 1])
#         # Shuffle each batch's indices uniquely
#         batched_indices = tf.map_fn(lambda x: tf.random.shuffle(x, seed = seed), indices, dtype=tf.int32)
#     else:
#         # Standard case handling (repeat shuffling logic you need for num_samples != batch_size)
#         full_indices = tf.range(num_samples, dtype=tf.int32)
#         shuffled_indices = tf.random.shuffle(full_indices, seed = seed)
#         if batch_size * num_batches > num_samples:
#             multiples = (batch_size * num_batches // num_samples) + 1
#             shuffled_indices = tf.tile(shuffled_indices, [multiples])
#         batched_indices = tf.reshape(shuffled_indices[:batch_size * num_batches], [num_batches, batch_size])
    
#     return batched_indices


class NPLMMetric(TwoSampleTestBase):
    """
    """
    def __init__(self, 
                 data_input: TwoSampleTestInputs,
                 progress_bar: bool = False,
                 verbose: bool = False,
                 **nplm_kwargs
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
        self.nplm_kwargs = nplm_kwargs # Use the setter to validate arguments
        
        super().__init__(data_input = data_input, 
                         progress_bar = progress_bar,
                         verbose = verbose)
        
    @property
    def nplm_kwargs(self) -> Dict[str, Any]:
        return self._nplm_kwargs
    
    @nplm_kwargs.setter
    def nplm_kwargs(self, nplm_kwargs: Dict[str, Any]) -> None:
        valid_keys: Set[str] = {'M', 'lam', 'iter_list', 'weight'}

        for key in nplm_kwargs.keys():
            if key not in valid_keys:
                raise ValueError(f"Invalid key: {key}. Valid keys are {valid_keys}")

        # You can add more specific validations for each argument here
        if 'M' in nplm_kwargs:
            M_number = nplm_kwargs['M']
            if not isinstance(M_number, int) or M_number <= 0:
                raise ValueError("M must be a positive integer")
        else:
            raise ValueError("Missing required key: M")
            
        if 'lam' in nplm_kwargs:
            lam_value = nplm_kwargs['lam']
            if lam_value <= 0:
                raise ValueError("lam must be positive")
        else:
            raise ValueError("Missing required key: lam")
        
        if 'weight' in nplm_kwargs:
            weight_value = nplm_kwargs['weight']
            if weight_value <= 0:
                raise ValueError("weight must be positive")
        else:
            raise ValueError("Missing required key: weight")
        
        if 'iter_list' in nplm_kwargs:
            iterations = nplm_kwargs['iter_list']
            #if iterations <= 0:
                #raise ValueError("iter_list must be positive")
        else:
            raise ValueError("Missing required key: iter_list")
        
        self._nplm_kwargs = nplm_kwargs
    
        print(f"The weight (ratio between the number of points in data sample and reference sample) is: {weight_value}\n"
              f"The number of Nyström center is: {M_number}\n"
              f"The lambda value is: {lam_value}\n"
              f"The number of iterations is: {iterations}")

    def compute(self) -> None:
        """
        Function that computes the NPLM  metric from two multivariate samples
        selecting always the Test_np method. This is because Falkon cannot use tf_vectors. 
        
        Parameters:
        ----------

        Returns:
        -------
        None
        """
        if self.use_tf:
            print("Warning: TensorFlow cannot be used with Falkon. Falkon wants PyTorch arrays. Using NumPy instead.")
            self.Test_np()
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

        ref_sample_for_sigma = dist_1_symb.sample(10000) #set_dist_num_from_symb(dist = dist_1_symb, nsamples = 10000, dtype = dtype)  
        flk_sigma = candidate_sigma(ref_sample_for_sigma, perc=90)
        print(f"The gaussian kernel sigma is estimated as the 90th percentile of the pairwise distance among 10000 points extracted from the reference distribution.\n"
              f"Its value is: {flk_sigma}")

        flk_config = get_logflk_config(self._nplm_kwargs.get('M'), flk_sigma, self._nplm_kwargs.get('lam'), self._nplm_kwargs.get('weight'), self._nplm_kwargs.get('iter_list'), seed=None, cpu=False) #where do I get M, lam, weight?
        flk_config['seed'] = 0 # seed for center selection

        def start_calculation() -> None:
            conditional_print(self.verbose, "\n------------------------------------------")
            conditional_print(self.verbose, "Starting nplm metric calculation...")
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
        #metric_error_list: List[float] = []

        start_calculation()
        init_progress_bar()
            
        self.Inputs.reset_seed_generator()
        
        conditional_print(self.verbose, "Running numpy NPLM calculation...")
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
            
            X_k = np.concatenate((dist_1_k, dist_2_k), axis=0)            # create the features array
            Y_k = np.zeros(shape=(dist_1_k.shape[0]+dist_2_k.shape[0],1)) # assign lables
            Y_k[dist_1_k.shape[0]:,:] = np.ones((dist_2_k.shape[0],1))    # flip labels to one for data
            
            print(f"The shape of X is: {X_k.shape}\n"
                  f"The shape of X is: {Y_k.shape}")

            preds_k = trainer(X_k, Y_k, flk_config)

            metric: float
            #metric_means: float
            #metric_stds: float

            metric = compute_t(preds_k, Y_k, self._nplm_kwargs.get('weight'))
            metric_list.append(metric)
            #metric_means.append(np.mean(list1)) # type: ignore
            #metric_stds.append(np.std(list1)) # type: ignore

            update_progress_bar()
        
        close_progress_bar()
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "NPLM Test"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "numpy"}}
        result_value: Dict[str, Optional[DataTypeNP]] = {"metric_list": np.array(metric_list)}
                                                         #,"metric_error_list": np.array(metric_error_list)}
        result: TwoSampleTestResult = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
        self.Results.append(result)

    def Test_tf(self, max_vectorize: int = 100) -> None:    
        raise NotImplementedError("You cannot use Tensorflow with Falkon. Please swap to Test.np().")
    
    # def Test_tf(self, max_vectorize: int = 100) -> None:
    #     """
    #     Function that computes the FGD  metric (and its uncertainty) from two multivariate samples
    #     using tensorflow functions.
    #     The calculation is performed in batches of size batch_size.
    #     The number of batches is niter.
    #     The total number of samples is niter*batch_size.
    #     The calculation is parallelized over max_vectorize (out of niter).
    #     The results are stored in the Results attribute.

    #     Parameters:
    #     ----------
    #     max_vectorize: int, optional, default = 100
    #         A maximum number of batch_size*max_vectorize samples per time are processed by the tensorflow backend.
    #         Given a value of max_vectorize, the niter FGD calculations are split in chunks of max_vectorize.
    #         Each chunk is processed by the tensorflow backend in parallel. If ndims is larger than max_vectorize,
    #         the calculation is vectorized niter times over ndims.

    #     Returns:
    #     --------
    #     None
    #     """
    #     max_vectorize = int(max_vectorize)
    #     # Set alias for inputs
    #     if isinstance(self.Inputs.dist_1_num, np.ndarray):
    #         dist_1_num: tf.Tensor = tf.convert_to_tensor(self.Inputs.dist_1_num)
    #     else:
    #         dist_1_num = self.Inputs.dist_1_num # type: ignore
    #     if isinstance(self.Inputs.dist_2_num, np.ndarray):
    #         dist_2_num: tf.Tensor = tf.convert_to_tensor(self.Inputs.dist_2_num)
    #     else:
    #         dist_2_num = self.Inputs.dist_2_num # type: ignore
    #     if isinstance(self.Inputs.dist_1_symb, tfp.distributions.Distribution):
    #         dist_1_symb: tfp.distributions.Distribution = self.Inputs.dist_1_symb
    #     else:
    #         raise TypeError("dist_1_symb must be a tfp.distributions.Distribution object when use_tf is True.")
    #     if isinstance(self.Inputs.dist_2_symb, tfp.distributions.Distribution):
    #         dist_2_symb: tfp.distributions.Distribution = self.Inputs.dist_2_symb
    #     else:
    #         raise TypeError("dist_2_symb must be a tfp.distributions.Distribution object when use_tf is True.")
    #     ndims: int = self.Inputs.ndims
    #     niter: int
    #     batch_size: int
    #     niter, batch_size = [int(i) for i in self.get_niter_batch_size_tf()] # type: ignore
    #     dtype: tf.DType = tf.as_dtype(self.Inputs.dtype)
        
    #     # Utility functions
    #     def start_calculation() -> None:
    #         conditional_tf_print(self.verbose, "\n------------------------------------------")
    #         conditional_tf_print(self.verbose, "Starting FGD metric calculation...")
    #         conditional_tf_print(self.verbose, "Running TF FGD calculation...")
    #         conditional_tf_print(self.verbose, "niter =", niter)
    #         conditional_tf_print(self.verbose, "batch_size =", batch_size)
    #         self._start = timer()

    #     def end_calculation() -> None:
    #         self._end = timer()
    #         elapsed = self.end - self.start
    #         conditional_tf_print(self.verbose, "FGD metric calculation completed in", str(elapsed), "seconds.")
                    
    #     def set_dist_num_from_symb(dist: DistTypeTF,
    #                                nsamples: int,
    #                               ) -> tf.Tensor:
    #         dist_num: tf.Tensor = generate_and_clean_data(dist, nsamples, self.Inputs.batch_size_gen, dtype = self.Inputs.dtype, seed_generator = self.Inputs.seed_generator, strategy = self.Inputs.strategy) # type: ignore
    #         return dist_num
        
    #     def return_dist_num(dist_num: tf.Tensor) -> tf.Tensor:
    #         return dist_num
        
    #     @tf.function(jit_compile=False, reduce_retracing=True)
    #     def batched_test_sub(dist_1_k_replica: tf.Tensor, 
    #                          dist_2_k_replica: tf.Tensor
    #                         ) -> DataTypeTF:
    #         def loop_body(idx):
    #             vals, batches = fgd_tf(dist_1_k_replica[idx, :, :], dist_2_k_replica[idx, :, :], **self.fgd_kwargs) # type: ignore
    #             vals = tf.cast(vals, dtype=dtype)
    #             batches = tf.cast(batches, dtype=dtype)
    #             return vals, batches

    #         # Vectorize over ndims*chunk_size
    #         vals_list: tf.Tensor
    #         batches_list: tf.Tensor
    #         vals_list, batches_list = tf.vectorized_map(loop_body, tf.range(tf.shape(dist_1_k_replica)[0])) # type: ignore
            
    #         res: DataTypeTF = tf.concat([vals_list, batches_list], axis=1) # type: ignore
    #         return res
        
    #     #@tf.function(jit_compile=False, reduce_retracing=True)
    #     def batched_test(start: tf.Tensor, 
    #                      end: tf.Tensor
    #                     ) -> DataTypeTF:
    #         # Define batched distributions
    #         dist_1_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
    #                                            true_fn = lambda: set_dist_num_from_symb(dist_1_symb, nsamples = batch_size * (end - start)), # type: ignore
    #                                            false_fn = lambda: return_dist_num(dist_1_num[start * batch_size : end * batch_size, :])) # type: ignore
    #         dist_2_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
    #                                            true_fn = lambda: set_dist_num_from_symb(dist_2_symb, nsamples = batch_size * (end - start)), # type: ignore
    #                                            false_fn = lambda: return_dist_num(dist_2_num[start * batch_size : end * batch_size, :])) # type: ignore

    #         dist_1_k = tf.reshape(dist_1_k, (end - start, batch_size, ndims)) # type: ignore
    #         dist_2_k = tf.reshape(dist_2_k, (end - start, batch_size, ndims)) # type: ignore

    #         res: DataTypeTF = batched_test_sub(dist_1_k, dist_2_k) # type: ignore
    
    #         return res
        
    #     def compute_test(max_vectorize: int = 100) -> Tuple[DataTypeTF, tf.Tensor]:
    #         # Check if numerical distributions are empty and print a warning if so
    #         conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_1_num tensor is empty. Batches will be generated 'on-the-fly' from dist_1_symb.") # type: ignore
    #         conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_2_num tensor is empty. Batches will be generated 'on-the-fly' from dist_2_symb.") # type: ignore
            
    #         # Ensure that max_vectorize is an integer larger than ndims
    #         max_vectorize = int(tf.cast(tf.minimum(max_vectorize, niter),tf.int32)) # type: ignore

    #         # Compute the maximum number of iterations per chunk
    #         max_iter_per_chunk: int = max_vectorize # type: ignore
            
    #         # Compute the number of chunks
    #         nchunks: int = int(tf.cast(tf.math.ceil(niter / max_iter_per_chunk), tf.int32)) # type: ignore
    #         conditional_tf_print(tf.logical_and(self.verbose,tf.logical_not(tf.equal(nchunks,1))), "nchunks =", nchunks) # type: ignore

    #         res: tf.TensorArray = tf.TensorArray(dtype, size = nchunks)
    #         res_vals: tf.TensorArray = tf.TensorArray(dtype, size = nchunks)
    #         res_batches: tf.TensorArray = tf.TensorArray(dtype, size = nchunks)

    #         def body(i: int, 
    #                  res: tf.TensorArray
    #                 ) -> Tuple[int, tf.TensorArray]:
    #             start: tf.Tensor = tf.cast(i * max_iter_per_chunk, tf.int32) # type: ignore
    #             end: tf.Tensor = tf.cast(tf.minimum(start + max_iter_per_chunk, niter), tf.int32) # type: ignore
    #             conditional_tf_print(tf.logical_and(tf.logical_or(tf.math.logical_not(tf.equal(start,0)),tf.math.logical_not(tf.equal(end,niter))), self.verbose), "Iterating from", start, "to", end, "out of", niter, ".") # type: ignore
    #             chunk_result: DataTypeTF = batched_test(start, end) # type: ignore
    #             res = res.write(i, chunk_result)
    #             return i+1, res
            
    #         def cond(i: int, 
    #                  res: tf.TensorArray):
    #             return i < nchunks
            
    #         _, res = tf.while_loop(cond, body, [0, res])
            
    #         for i in range(nchunks):
    #             res_i: DataTypeTF = tf.convert_to_tensor(res.read(i))
    #             npoints: tf.Tensor = res_i.shape[1] // 2 # type: ignore
    #             res_vals = res_vals.write(i, res_i[:, :npoints]) # type: ignore
    #             res_batches = res_batches.write(i, res_i[:, npoints:]) # type: ignore
                
    #         vals_list: DataTypeTF = res_vals.stack() # type: ignore
    #         batches_list: tf.Tensor = res_batches.stack() # type: ignore
            
    #         shape = tf.shape(vals_list)
    #         vals_list = tf.reshape(vals_list, (shape[0] * shape[1], shape[2]))
    #         batches_list = tf.reshape(batches_list, (shape[0] * shape[1], shape[2]))
    #         #vals_list: DataTypeTF = tf.squeeze(res_vals.stack())
    #         #batches_list: tf.Tensor = tf.squeeze(res_batches.stack())
            
    #         # Flatten vals_list and batches_list to 1-D arrays
    #         #vals_list = tf.reshape(vals_list, [-1])  # Flatten to 1-D
    #         #batches_list = tf.reshape(batches_list, [-1])  # Flatten to 1-D

    #         return vals_list, batches_list

    #     start_calculation()
        
    #     self.Inputs.reset_seed_generator()
        
    #     vals_list: DataTypeTF
    #     batches_list: tf.Tensor
    #     vals_list, batches_list  = compute_test(max_vectorize = max_vectorize)
                
    #     #print(f"vals_list: {vals_list=}")
    #     #print(f"batches_list: {batches_list=}")
        
    #     metric_list: DataTypeNP
    #     metric_error_list: DataTypeNP
    #     metric_list, metric_error_list = fgd_tf_fit(vals_list, batches_list)
                             
    #     end_calculation()
        
    #     timestamp: str = datetime.now().isoformat()
    #     test_name: str = "FGD Test_tf"
    #     parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "tensorflow"}}
    #     result_value: Dict[str, Optional[DataTypeNP]] = {"metric_list": metric_list,
    #                                                      "metric_error_list": metric_error_list}
    #     result: TwoSampleTestResult = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
    #     self.Results.append(result)