__all__ = [
    'TwoSampleTestInputs',
    'TwoSampleTestResult',
    'TwoSampleTestBase'
]

from abc import ABC, abstractmethod
import numpy as np
import json
import pandas as pd # type: ignore
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import traceback
from timeit import default_timer as timer
from tqdm import tqdm # type: ignore
from . import utils
from .utils import reset_random_seeds
from .utils import parse_input_dist_np
from .utils import parse_input_dist_tf
from .utils import get_best_dtype_np
from .utils import get_best_dtype_tf
from .utils import generate_and_clean_data
from .utils import NumpyDistribution
from dataclasses import dataclass

from typing import Tuple, Union, Optional, Type, Dict, Any, List
from .utils import DTypeType, IntTensor, FloatTensor, BoolTypeTF, BoolTypeNP, IntType, DataTypeTF, DataTypeNP, DataType, DistTypeTF, DistTypeNP, DistType, DataDistTypeNP, DataDistTypeTF, DataDistType, BoolType

class TwoSampleTestInputs(object):
    """
    Class for validating data.
    """
    def __init__(self, 
                 dist_1_input: DataDistType,
                 dist_2_input: DataDistType,
                 niter: int = 10,
                 batch_size_test: int = 100_000,
                 batch_size_gen: int = 1_000,
                 small_sample_threshold: float = 1e7,
                 dtype_input: DTypeType = np.float32,
                 seed_input: Optional[int] = None,
                 use_tf: bool = False,
                 mirror_strategy: bool = False,
                 verbose: bool = False,
                ) -> None:
        # Attributes from arguments
        self._dist_1_input: DataDistType = dist_1_input
        self._dist_2_input: DataDistType = dist_2_input
        self._niter: int = niter
        self._batch_size_test: int = batch_size_test
        self._batch_size_gen: int = batch_size_gen
        self._small_sample_threshold: int = int(small_sample_threshold)
        self._dtype_input: tf.DType = dtype_input
        self._seed: int = seed_input if seed_input is not None else int(np.random.randint(0, 2**32 - 1))
        self._use_tf: bool = use_tf
        self.mirror_strategy = mirror_strategy # Setting using the setter to ensure that also self._strategy gets properly defined
        self.verbose: bool = verbose
        
        # Attributes from preprocessing
        self._is_symb_1: BoolType
        self._is_symb_2: BoolType
        self._dist_1_symb: DistType
        self._dist_2_symb: DistType
        self._dist_1_num: DataType
        self._dist_2_num: DataType
        self._ndims_1: IntType
        self._ndims_2: IntType
        self._ndims: int
        self._nsamples_1: IntType
        self._nsamples_2: IntType
        self._nsamples: int
        self._dtype: DTypeType
        self._small_sample: bool
        self._seed_generator: tf.random.Generator
        self._strategy: Optional[tf.distribute.Strategy]
        
        # Preprocessing
        self.__preprocess(verbose = verbose)
            
    @property
    def dist_1_input(self) -> DataDistType:
        return self._dist_1_input
    
    @dist_1_input.setter
    def dist_1_input(self, 
                     dist_1_input: DataDistType
                    ) -> None:
        if isinstance(dist_1_input, (np.ndarray, NumpyDistribution, tf.Tensor, tfd.Distribution)):
            self._dist_1_input = dist_1_input
        else:
            raise TypeError("dist_1_input must be a np.ndarray, NumpyDistribution, tf.Tensor, or tfd.Distribution")
        self.__preprocess(verbose = False)
        
    @property
    def dist_2_input(self) -> DataDistType:
        return self._dist_2_input

    @dist_2_input.setter
    def dist_2_input(self,
                     dist_2_input: DataDistType
                    ) -> None:
        if isinstance(dist_2_input, (np.ndarray, NumpyDistribution, tf.Tensor, tfd.Distribution)):
            self._dist_2_input = dist_2_input
        else:   
            raise TypeError("dist_2_input must be a np.ndarray, NumpyDistribution, tf.Tensor, or tfd.Distribution")
        self.__preprocess(verbose = False)
            
    @property
    def niter(self) -> int:
        return self._niter
            
    @niter.setter
    def niter(self,
              niter: int
              ) -> None:
        if isinstance(niter, int):
            if niter > 0:
                self._niter = niter
            else:
                raise ValueError("niter must be positive")
        else:
            raise TypeError("niter must be an int")
        self.__preprocess(verbose = False)

    @property
    def batch_size_test(self) -> int:
        return self._batch_size_test
    
    @batch_size_test.setter
    def batch_size_test(self,
                        batch_size_test: int
                       ) -> None:
        if isinstance(batch_size_test, int):
            if batch_size_test > 0:
                self._batch_size_test = batch_size_test
            else:
                raise ValueError("batch_size_test must be positive")
        self.__preprocess(verbose = False)
        
    @property
    def batch_size_gen(self) -> int:
        return self._batch_size_gen
    
    @batch_size_gen.setter
    def batch_size_gen(self,
                       batch_size_gen: int
                      ) -> None:
        if isinstance(batch_size_gen, int):
            if batch_size_gen > 0:
                self._batch_size_gen = batch_size_gen
            else:
                raise ValueError("batch_size_gen must be positive")
            
    @property
    def small_sample_threshold(self) -> int:
        return self._small_sample_threshold
    
    @small_sample_threshold.setter
    def small_sample_threshold(self,
                               small_sample_threshold: int
                              ) -> None:
          if isinstance(small_sample_threshold, int):
                if small_sample_threshold > 0:
                    self._small_sample_threshold = small_sample_threshold
                else:
                    raise ValueError("small_sample_threshold must be positive")
          self.__preprocess(verbose = False)
        
    @property
    def dtype_input(self) -> Union[tf.DType, np.dtype, type]:
        return self._dtype_input

    @dtype_input.setter
    def dtype_input(self,
                    dtype_input: Union[tf.DType, np.dtype, type]
                   ) -> None:
        if isinstance(dtype_input, (tf.DType, np.dtype, type)):
            self._dtype_input = dtype_input
        else:
            raise TypeError("dtype_input must be a tf.DType, np.dtype, or type")
        self.__preprocess(verbose = False)
        
    @property
    def seed(self) -> int:
        return self._seed
    
    @seed.setter
    def seed(self,
             seed_input: int
            ) -> None:
        if isinstance(seed_input, int):
            self._seed = seed_input
        else:
            raise TypeError("seed_input must be an int")
        self.__preprocess(verbose = False)
        
    @property
    def seed_generator(self) -> tf.random.Generator:
        return self._seed_generator
        
    @property
    def use_tf(self) -> bool:
        return self._use_tf

    @use_tf.setter
    def use_tf(self,
                use_tf: BoolType
                ) -> None:
        if isinstance(use_tf, (bool,np.bool_)):
            self._use_tf = bool(use_tf)
        elif isinstance(use_tf, tf.Tensor):
            if isinstance(use_tf.numpy(), np.bool_):
                self._use_tf = bool(use_tf)
            else:
                raise TypeError("use_tf must be a bool or tf.Tensor with numpy() method returning a np.bool_")
        else:
            raise TypeError("use_tf must be a bool or tf.Tensor with numpy() method returning a np.bool_")
        self.__preprocess(verbose = False)
        
    @property
    def mirror_strategy(self) -> bool:
        return self._mirror_strategy
    
    @mirror_strategy.setter
    def mirror_strategy(self,
                        mirror_strategy: BoolType
                          ) -> None:
        if isinstance(mirror_strategy, (bool,np.bool_)):
            self._mirror_strategy = bool(mirror_strategy)
        elif isinstance(mirror_strategy, tf.Tensor):
            if isinstance(mirror_strategy.numpy(), np.bool_):
                self._mirror_strategy = bool(mirror_strategy)
            else:
                raise TypeError("mirror_strategy must be a bool or tf.Tensor with numpy() method returning a np.bool_")
        if self._mirror_strategy:
            self._strategy = tf.distribute.MirroredStrategy()
        else:
            self._strategy = None
            
    @property
    def strategy(self) -> Optional[tf.distribute.Strategy]:
        return self._strategy
        
    @property
    def verbose(self) -> bool: # type: ignore
        return self._verbose

    @verbose.setter
    def verbose(self, # type: ignore
                verbose: Union[int,bool]
                ) -> None:
        if isinstance(verbose, bool):
            self._verbose = verbose
        elif isinstance(verbose, int):
            self._verbose = bool(verbose)
        else:
            raise TypeError("verbose must be a bool or an int (which is automatically converted to a bool)")
            
    @property
    def is_symb_1(self) -> BoolType:
        return self._is_symb_1
    
    @property
    def is_symb_2(self) -> BoolType:
        return self._is_symb_2
            
    @property
    def dist_1_symb(self) -> DistType:
        return self._dist_1_symb
    
    @property
    def dist_2_symb(self) -> DistType:
        return self._dist_2_symb
    
    @property
    def dist_1_num(self) -> DataType:
        return self._dist_1_num

    @property
    def dist_2_num(self) -> DataType:
        return self._dist_2_num
    
    @property
    def ndims_1(self) -> IntType:
        return self._ndims_1
    
    @property
    def ndims_2(self) -> IntType:
        return self._ndims_2
    
    @property
    def ndims(self) -> int:
        return self._ndims

    @property
    def nsamples_1(self) -> IntType:
        return self._nsamples_1

    @property
    def nsamples_2(self) -> IntType:
        return self._nsamples_2
    
    @property
    def nsamples(self) -> int:
        return self._nsamples
        
    @property
    def dtype_1(self) -> DTypeType:
        try:
            if isinstance(self.dist_1_num, (np.ndarray,tf.Tensor)):
                return self.dist_1_num.dtype
            else:
                raise AttributeError("dist_1_num should be a np.ndarray or tf.Tensor")
        except AttributeError:
            raise AttributeError("dist_1_num should be a np.ndarray or tf.Tensor")
            
    @property
    def dtype_2(self) -> DTypeType:
        try:
            if isinstance(self.dist_2_num, (np.ndarray,tf.Tensor)):
                return self.dist_2_num.dtype
            else:
                raise AttributeError("dist_2_num should be a np.ndarray or tf.Tensor")
        except AttributeError:
            raise AttributeError("dist_2_num should be a np.ndarray or tf.Tensor")

    @property
    def dtype(self) -> DTypeType:
        return self._dtype
    
    @property
    def dtype_str(self) -> str:
        if isinstance(self.dtype, (tf.DType,np.dtype)):
            return self.dtype.name
        else:
            raise TypeError("dtype is not of type tf.DType or np.dtype")
    
    @property
    def small_sample(self) -> bool:
        return self._small_sample
    
    @small_sample.setter
    def small_sample(self,
                     small_sample: BoolType
                    ) -> None:
        if isinstance(small_sample, (bool,np.bool_)):
            self._small_sample = bool(small_sample)
        elif isinstance(small_sample, tf.Tensor):
            if isinstance(small_sample.numpy(), np.bool_):
                self._small_sample = bool(small_sample)
            else:
                raise TypeError("small_sample must be a bool or tf.Tensor with numpy() method returning a np.bool_")
        else:
            raise TypeError("small_sample must be a bool or tf.Tensor with numpy() method returning a np.bool_")
        self.__check_set_distributions()
        if self.small_sample:
            if not (self.batch_size_test*self.niter*self.ndims <= self.small_sample_threshold or self.nsamples*self.ndims <= self.small_sample_threshold):
                print("Warning: small_sample is set to True, but the number of samples is large. This may cause memory issues.")
        if not self.small_sample:
            if self.batch_size_test*self.niter*self.ndims <= self.small_sample_threshold or self.nsamples*self.ndims <= self.small_sample_threshold:
                print("Warning: small_sample is set to False, but the number of samples is small. Setting small_sample to True may speed up calculations.")
                        
    def __parse_input_dist(self,
                           dist_input: DataDistType,
                           verbose: bool = False
                          ) -> Tuple[BoolType, DistType, DataType, IntType, IntType]:
        if isinstance(dist_input, NumpyDistribution):
            if self.use_tf:
                if self.verbose:
                    print("To use tf mode, please use tf distributions or numerical tensors/arrays.")
                self.use_tf = False
        if self.use_tf:
            if isinstance(dist_input, (tf.Tensor, tfp.distributions.Distribution)):
                return parse_input_dist_tf(dist_input = dist_input, verbose = verbose)
            elif isinstance(dist_input, (np.ndarray, NumpyDistribution)):
                if self.verbose:
                    print("To use tf mode, please use tf distributions or numerical tensors/arrays.")
                self.use_tf = False
                return parse_input_dist_np(dist_input = dist_input, verbose = verbose)
            else:
                raise TypeError("dist_input must be a tf.Tensor, tfp.distributions.Distribution, np.ndarray, or NumpyDistribution")
        else:
            if isinstance(dist_input, (tf.Tensor, tfp.distributions.Distribution)):
                if self.verbose:
                    print("Using numpy mode with TensorFlow inputs.")
                is_symb, dist_symb, dist_num, ndims, nsamples = parse_input_dist_tf(dist_input = dist_input, verbose = verbose)
                is_symb = bool(is_symb)
                dist_num = dist_num.numpy() # type: ignore
                ndims = int(ndims) # type: ignore
                nsamples = int(nsamples) # type: ignore
                return is_symb, dist_symb, dist_num, ndims, nsamples
            elif isinstance(dist_input, (np.ndarray, NumpyDistribution)):
                return parse_input_dist_np(dist_input = dist_input, verbose = verbose)
            else:
                raise TypeError("dist_input must be a tf.Tensor, tfp.distributions.Distribution, np.ndarray, or NumpyDistribution")
        
    def __get_best_dtype(self,
                         dtype_1: DTypeType,
                         dtype_2: DTypeType,
                        ) -> DTypeType:   
        if self.use_tf:
            dtype_1 = tf.as_dtype(dtype_1)
            dtype_2 = tf.as_dtype(dtype_2)
            return get_best_dtype_tf(dtype_1 = dtype_1, dtype_2 = dtype_2)
        else:
            if isinstance(dtype_1, tf.DType):
                dtype_1 = dtype_1.as_numpy_dtype
            if isinstance(dtype_2, tf.DType):
                dtype_2 = dtype_2.as_numpy_dtype
            return get_best_dtype_np(dtype_1 = dtype_1, dtype_2 = dtype_2)
        
    def __parse_input_distributions(self,
                                    verbose: bool = False
                                    ) -> None:
        self._is_symb_1, self._dist_1_symb, self._dist_1_num, self._ndims_1, self._nsamples_1 = self.__parse_input_dist(dist_input = self._dist_1_input, verbose = verbose)
        self._is_symb_2, self._dist_2_symb, self._dist_2_num, self._ndims_2, self._nsamples_2 = self.__parse_input_dist(dist_input = self._dist_2_input, verbose = verbose)
            
    def __check_set_dtype(self) -> None:
        self._dtype = self.__get_best_dtype(self.dtype_input, self.__get_best_dtype(self.dtype_1, self.dtype_2))
    
    def __check_set_ndims_np(self) -> None:
        self._ndims_1 = int(self.ndims_1) # type: ignore
        self._ndims_2 = int(self.ndims_2) # type: ignore 
        if not (isinstance(self.ndims_1, int) and isinstance(self.ndims_2, int)):
            raise ValueError("ndims_1 and ndims_2 should be integers when in 'numpy' mode.")
        if self.ndims_1 != self.ndims_2:
            raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
        else:
            self._ndims = self.ndims_1
            
    def __check_set_ndims_tf(self) -> None:
        # Utility functions
        def set_ndims(value: IntType) -> None:
            self._ndims = int(tf.constant(value))

        def raise_non_integer(value: IntType) -> None:
            value_tf = tf.constant(value)
            tf.debugging.assert_equal(value_tf.dtype.is_integer, tf.constant(True), message="non integer dimensions")
            
        def raise_none_equal_dims_error(value1: Any, value2: Any) -> None:
            tf.debugging.assert_equal(value1, value2, message="dist_1 and dist_2 must have the same number of dimensions")
        
        raise_non_integer(self.ndims_1)
        raise_non_integer(self.ndims_2)
        tf.cond(tf.equal(self.ndims_1, self.ndims_2), 
                true_fn = lambda: set_ndims(self.ndims_1), 
                false_fn = lambda: raise_none_equal_dims_error(self.ndims_1, self.ndims_2))
            
    def __check_set_ndims(self) -> None:
        if self.use_tf:
            self.__check_set_ndims_tf()
        else:
            self.__check_set_ndims_np()
            
    def __check_set_nsamples_np(self) -> None:
        if not (isinstance(self.nsamples_1, int) and isinstance(self.nsamples_2, int)):
            raise ValueError("nsamples_1 and nsamples_2 should be integers when in 'numpy' mode.")
        if self.nsamples_1 != 0 and self.nsamples_2 != 0:
            self._nsamples = np.minimum(int(self.nsamples_1), int(self.nsamples_2))
        elif self.nsamples_1 != 0 and self.nsamples_2 == 0:
            self._nsamples = int(self.nsamples_1)
        elif self.nsamples_1 == 0 and self.nsamples_2 != 0:
            self._nsamples = int(self.nsamples_2)
        elif self.nsamples_1 == 0 and self.nsamples_2 == 0:
            self._nsamples = int(self.batch_size_test * self.niter)
        else:
            raise ValueError("nsamples_1 and nsamples_2 should be positive integers or zero.")
        
    def __check_set_nsamples_tf(self) -> None:
        # Utility functions
        def set_nsamples(value: IntType) -> None:
            self._nsamples = int(tf.constant(value))
        nsamples_1 = int(tf.constant(self.nsamples_1))
        nsamples_2 = int(tf.constant(self.nsamples_2))
        nsamples_min = int(tf.constant(tf.minimum(nsamples_1, nsamples_2)))
        tf.cond(tf.not_equal(self.nsamples_1, tf.constant(0)),
                true_fn = lambda: tf.cond(tf.not_equal(self.nsamples_2, tf.constant(0)),
                                          true_fn = lambda: set_nsamples(nsamples_min),
                                          false_fn = lambda: set_nsamples(self.nsamples_1)),
                false_fn = lambda: tf.cond(tf.not_equal(self.nsamples_2, tf.constant(0)),
                                           true_fn = lambda: set_nsamples(self.nsamples_2),
                                           false_fn = lambda: set_nsamples(self.batch_size_test * self.niter)))
        
    def __check_set_nsamples(self) -> None:
        if self.use_tf:
            self.__check_set_nsamples_tf()
        else:
            self.__check_set_nsamples_np()
            
    def __check_set_small_sample_np(self) -> None:
        if self.batch_size_test * self.niter * self.ndims <= self.small_sample_threshold or self.nsamples * self.ndims <= self.small_sample_threshold:
            self._small_sample = True
        else:
            self._small_sample = False
    
    def __check_set_small_sample_tf(self) -> None:
        # Utility functions
        def set_small_sample(value: bool) -> None:
            self._small_sample = value
        #print(f"batch_size_test: {self.batch_size_test}")
        #print(f"niter: {self.niter}")
        #print(f"ndims: {self.ndims}")
        #print(f"nsamples: {self.nsamples}")
        #print(f"small_sample_threshold: {self.small_sample_threshold}")
        tf.cond(tf.logical_or(tf.less_equal(self.batch_size_test * self.niter * self.ndims, self.small_sample_threshold), 
                              tf.less_equal(self.nsamples * self.ndims, self.small_sample_threshold)),
                true_fn = lambda: set_small_sample(True),
                false_fn = lambda: set_small_sample(False))
        #print(f"small_sample: {self.small_sample}")
        
    def __check_set_small_sample(self) -> None:
        if self.use_tf:
            self.__check_set_small_sample_tf()
        else:
            self.__check_set_small_sample_np()
            
    def __check_set_distributions_np(self) -> None:
        # Reset random seeds
        if self.is_symb_1:
            if self.small_sample:
                if isinstance(self.dist_1_symb, NumpyDistribution):
                    seed_dist_1 = int(self.seed_generator.make_seeds()[0,0].numpy()) # type: ignore
                    self._dist_1_num = self.dist_1_symb.sample(self.nsamples, seed = int(seed_dist_1)).astype(self.dtype)
                elif isinstance(self._dist_1_symb, tfp.distributions.Distribution):
                    if self.is_symb_2 and isinstance(self.dist_2_symb, tfp.distributions.Distribution):
                        pass
                    else:
                        if self.verbose:
                            print("Generating dist_1_num with numpy function.")
                        seed_dist_1 = int(self.seed_generator.make_seeds()[0,0].numpy()) # type: ignore
                        self._dist_1_num = self._dist_1_symb.sample(self.nsamples, seed = int(seed_dist_1)).numpy().astype(self.dtype) # type: ignore
                else:
                    raise ValueError("dist_1_symb should be a subclass of NumpyDistribution or tfp.distributions.Distribution.")
            else:
                self._dist_1_num = tf.convert_to_tensor([[]], dtype = self.dist_1_symb.dtype) # type: ignore
        else:
            if isinstance(self.dist_1_num, (np.ndarray, tf.Tensor)):
                self._dist_1_num = self.dist_1_num[:self.nsamples,:].astype(self.dtype)
            else:
                raise ValueError("dist_1_num should be an instance of np.ndarray or tf.Tensor.")
        if self.is_symb_2:
            if self.small_sample:
                if isinstance(self.dist_2_symb, NumpyDistribution):
                    seed_dist_2 = int(self.seed_generator.make_seeds()[0,0].numpy()) # type: ignore
                    self._dist_2_num = self.dist_2_symb.sample(self.nsamples, seed = int(seed_dist_2)).astype(self.dtype)
                elif isinstance(self.dist_2_symb, tfp.distributions.Distribution):
                    if self.is_symb_1 and isinstance(self.dist_1_symb, tfp.distributions.Distribution):
                        pass
                    else:
                        if self.verbose:
                            print("Generating dist_2_num with numpy function.")
                        seed_dist_2 = int(self.seed_generator.make_seeds()[0,0].numpy()) # type: ignore
                        self._dist_2_num = self._dist_2_symb.sample(self.nsamples, seed = int(seed_dist_2)).numpy().astype(self.dtype) # type: ignore
                else:
                    raise ValueError("dist_2_symb should be a subclass of NumpyDistribution or tfp.distributions.Distribution.")
            else:
                self._dist_2_num = tf.convert_to_tensor([[]], dtype = self.dist_2_symb.dtype) # type: ignore
        else:
            if isinstance(self.dist_2_num, (np.ndarray, tf.Tensor)):
                self._dist_2_num = self.dist_2_num[:self.nsamples,:].astype(self.dtype)
            else:  
                raise ValueError("dist_2_num should be an instance of np.ndarray or tf.Tensor.")
        if self.is_symb_1 and self.is_symb_2:
            if isinstance(self.dist_1_symb, tfp.distributions.Distribution) and isinstance(self.dist_2_symb, tfp.distributions.Distribution):
                if self.small_sample:
                    if self.verbose:
                        print("Generating dist_1_num and dist_2_num with tensorflow function.")
                    self.__check_set_distributions_tf()
                    self._dist_1_num = self.dist_1_num.numpy().astype(self.dtype) # type: ignore
                    self._dist_2_num = self.dist_2_num.numpy().astype(self.dtype) # type: ignore
                
            
    def __check_set_distributions_tf(self) -> None:
        # Utility functions
        def set_dist_num_from_symb(dist: tfp.distributions.Distribution) -> tf.Tensor:
            if self.verbose:
                print("Setting dist_num from dist_symb.")
            if isinstance(dist, tfp.distributions.Distribution):
                dist_num: tf.Tensor = generate_and_clean_data(dist, self.nsamples, self.batch_size_gen, dtype = self.dtype, seed_generator = self.seed_generator, strategy = self.strategy) # type: ignore
            else:
                raise ValueError("dist should be an instance of tfp.distributions.Distribution.")
            return dist_num
        
        def return_dist_num(dist_num: DataType) -> tf.Tensor:
            if self.verbose:
                print("Returning dist_num.")
            if isinstance(dist_num, tf.Tensor):
                return dist_num
            else:
                raise ValueError("dist_num should be an instance of tf.Tensor.")
            
        def reset_dist_num(dist: tfp.distributions.Distribution) -> tf.Tensor:
            if self.verbose:
                print("Resetting dist_num.")
            return tf.convert_to_tensor([[]], dtype = dist.dtype) # type: ignore
        
        if self.verbose:
            print("Checking and setting numerical distributions.")
        
        dist_1_num = tf.cond(self.is_symb_1,
                             true_fn = lambda: tf.cond(self.small_sample,
                                                       true_fn = lambda: set_dist_num_from_symb(self.dist_1_symb),
                                                       false_fn = lambda: reset_dist_num(self.dist_1_symb)),
                             false_fn = lambda: return_dist_num(self.dist_1_num))
        dist_2_num = tf.cond(self.is_symb_2,
                             true_fn = lambda: tf.cond(self.small_sample,
                                                       true_fn = lambda: set_dist_num_from_symb(self.dist_2_symb),
                                                       false_fn = lambda: reset_dist_num(self.dist_2_symb)),
                             false_fn = lambda: return_dist_num(self.dist_2_num))
        self._dist_1_num = tf.cast(dist_1_num, self.dtype)[:self.nsamples, :] # type: ignore
        self._dist_2_num = tf.cast(dist_2_num, self.dtype)[:self.nsamples, :] # type: ignore
        
    def __check_set_distributions(self) -> None:
        if self.use_tf:
            self.__check_set_distributions_tf()
        else:
            self.__check_set_distributions_np()
            
    def reset_seed_generator(self) -> None:
        # Reset seed and set seeds generator
        reset_random_seeds(seed = self.seed)
        self._seed_generator = tf.random.Generator.from_seed(self.seed)
                
    def __preprocess(self, 
                     verbose: bool = False) -> None:
        # Reset seed and set seeds generator
        self.reset_seed_generator()
        
        # Parse input distributions
        self.__parse_input_distributions(verbose = verbose)

        # Check and set dtype
        self.__check_set_dtype()
        
        # Check and set ndims
        self.__check_set_ndims()

        # Check and set nsamples
        self.__check_set_nsamples()
        
        # Check and set small sample
        self.__check_set_small_sample()

        # Check and set distributions
        self.__check_set_distributions() 
        
    @property
    def param_dict(self) -> Dict[str, Any]:
        return {"is_symb_1": bool(self.is_symb_1),
                "is_symb_2": bool(self.is_symb_2),
                "ndims": self.ndims,
                "niter": self.niter,
                "batch_size_test": self.batch_size_test,
                "batch_size_gen": self.batch_size_gen,
                "dtype": self.dtype_str,
                "small_sample_threshold": self.small_sample_threshold,
                "small_sample": self.small_sample}
    
@dataclass
class TwoSampleTestResult:
    def __init__(self,
                 timestamp: str,
                 test_name: str,
                 parameters: Dict[str, Any],
                 result_value: Dict[str, Optional[DataTypeNP]]
                ) -> None:
        self.timestamp: str = timestamp
        self.test_name: str = test_name
        self.__dict__.update(parameters)
        self.result_value: Dict[str, Optional[DataTypeNP]] = result_value
        for key, val in self.result_value.items():
            if isinstance(val, list):
                val = np.array(val)
    
    def result_to_dataframe(self):
        return pd.DataFrame.from_dict(self.__dict__, orient="index")

    def print_result(self,
                     print_mode: str = "full"):
        for k, v in self.__dict__.items():
            if print_mode == "full":
                print(f"{k}: {v}")
            elif print_mode == "parameters":
                if k != "result_value":
                    print(f"{k}: {v}")
            else:
                raise ValueError(f"print_mode must be either 'full' or 'parameters', but got {print_mode}")
            

class TwoSampleTestResults(object):
    def __init__(self) -> None:
        self._results: List[TwoSampleTestResult] = []
    
    @property
    def results(self) -> List[TwoSampleTestResult]:
        return self._results
        
    def append(self, item: TwoSampleTestResult) -> None:
        if isinstance(item, TwoSampleTestResult):
            self._results.append(item)
        else:
            raise ValueError('Can only add TwoSampleTestResult objects to the results.')

    def __getitem__(self, index: int) -> TwoSampleTestResult:
        return self.results[index]

    def __len__(self) -> int:
        return len(self.results)

    def __repr__(self) -> str:
        return repr(self.results)
    
    def print_results(self,
                      print_mode: str = "full"
                     ) -> None:
        for result in self.results:
            print("--------------------------------------------------------")  # add a blank line between results
            result.print_result(print_mode = print_mode)
    
    def get_results_as_dataframe(self,
                                 sort_kwargs: dict = {"by": ["batch_size_test","niter"], "ascending": [True]},
                                 print_mode: str = "full"
                                ) -> pd.DataFrame:
        df = pd.DataFrame()
        for result in self.results:
            df = pd.concat([df, result.result_to_dataframe().T])
        if print_mode == "full":
            df = df.sort_values(**sort_kwargs)
        elif print_mode == "parameters":
            df = df.drop(columns=["result_value"]).sort_values(**sort_kwargs)
        else:
            raise ValueError(f"print_mode must be either 'full' or 'parameters', but got {print_mode}")
        return df
    
    @property
    def results_df(self) -> pd.DataFrame:
        df = self.get_results_as_dataframe(print_mode = "full")
        return df
    
    @property
    def results_dict(self) -> Dict[str, Any]:
        return {str(k): v for k, v in self.results_df.to_dict(orient="index").items()}

    @property
    def results_params_df(self) -> pd.DataFrame:
        df = self.get_results_as_dataframe(print_mode = "parameters")
        return df
    
    @property
    def results_params_dict(self) -> Dict[str, Any]:
        return {str(k): v for k, v in self.results_params_df.to_dict(orient="index").items()}
    
    def save_to_json(self, filepath: str) -> None:
        results_data = [utils.convert_types_dict(result.__dict__) for result in self._results]
        with open(filepath, 'w') as file:
            json.dump(results_data, file, indent=4)

    def load_from_json(self, filepath: str):
        with open(filepath, 'r') as file:
            data = json.load(file)
        self._results = []
        for result_data in data:
            timestamp = result_data.pop("timestamp")
            test_name = result_data.pop("test_name")
            result_value_list = result_data.pop("result_value")
            result_value: Dict[str, Optional[DataTypeNP]] = {key: np.array(val) for key, val in result_value_list.items()}
            parameters = result_data
            result = TwoSampleTestResult(timestamp = timestamp,
                                         test_name = test_name,
                                         parameters = parameters,
                                         result_value = result_value)
            self._results.append(result)


class TwoSampleTestBase(ABC):
    """
    Base class for metrics.
    """
    def __init__(self, 
                 data_input: TwoSampleTestInputs,
                 progress_bar: bool = False,
                 verbose: bool = False
                ) -> None:
        self.Inputs: TwoSampleTestInputs = data_input
        self.progress_bar: bool = progress_bar
        self.verbose: bool = verbose
        self._start: float = 0.
        self._end: float = 0.
        self.pbar: tqdm = tqdm(disable=True)
        self._Results: TwoSampleTestResults = TwoSampleTestResults()
        
        
    @property
    def Inputs(self) -> TwoSampleTestInputs: # type: ignore
        return self._Inputs
    
    @Inputs.setter
    def Inputs(self, # type: ignore
                Inputs: TwoSampleTestInputs) -> None:
        if isinstance(Inputs, TwoSampleTestInputs):
            self._Inputs: TwoSampleTestInputs = Inputs
        else:
            raise TypeError(f"Inputs must be of type TwoSampleTestInputs, but got {type(Inputs)}")
        
    @property
    def progress_bar(self) -> bool: # type: ignore
        return self._progress_bar
    
    @progress_bar.setter
    def progress_bar(self, # type: ignore
                     progress_bar: bool) -> None:
        if isinstance(progress_bar, bool):
            self._progress_bar: bool = progress_bar
            #if self.Inputs.use_tf and self.progress_bar:
            #    self._progress_bar = False
            #    print("progress_bar is disabled when using tensorflow mode.")
        else:
            raise TypeError(f"progress_bar must be of type bool, but got {type(progress_bar)}")
        
    @property
    def verbose(self) -> bool: # type: ignore
        return self._verbose

    @verbose.setter
    def verbose(self, # type: ignore
                verbose: Union[int,bool]
                ) -> None:
        if isinstance(verbose, bool):
            self._verbose: bool = verbose
        elif isinstance(verbose, int):
            self._verbose = bool(verbose)
        else:
            raise TypeError("verbose must be a bool or an int (which is automatically converted to a bool)")
        
    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    @property
    def pbar(self) -> tqdm: # type: ignore
        return self._pbar
    
    @pbar.setter
    def pbar(self, # type: ignore
                pbar: tqdm) -> None:
        if isinstance(pbar, tqdm):
            self._pbar: tqdm = pbar
        else:
            raise TypeError(f"pbar must be of type tqdm, but got {type(pbar)}")
        
    @property
    def Results(self) -> TwoSampleTestResults:
        return self._Results
    
    @property
    def use_tf(self) -> bool:
        return self.Inputs.use_tf
    
    @property
    def small_sample(self) -> bool:
        return self.Inputs.small_sample
    
    @property
    def small_sample_threshold(self) -> int:
        return self.Inputs.small_sample_threshold
            
    def get_niter_batch_size_np(self) -> Tuple[int, int]:
        nsamples = self.Inputs.nsamples
        batch_size_test = self.Inputs.batch_size_test
        niter = self.Inputs.niter
        if nsamples < batch_size_test * niter:
            batch_size_test = nsamples // niter
        else:
            pass
        if batch_size_test == 0:
            raise ValueError("batch_size_test should be positive integer and number of samples should be larger than number of iterations.")
        return niter, batch_size_test
    
    def get_niter_batch_size_tf(self) -> Tuple[tf.Tensor, tf.Tensor]:
        nsamples: tf.Tensor = tf.cast(self.Inputs.nsamples, dtype = tf.int32) # type: ignore
        batch_size_test: tf.Tensor = tf.cast(self.Inputs.batch_size_test, dtype = tf.int32) # type: ignore
        niter: tf.Tensor = tf.cast(self.Inputs.niter, dtype = tf.int32) # type: ignore
        batch_size_test_tmp = tf.cond(nsamples < batch_size_test * niter, # type: ignore
                                    true_fn=lambda: nsamples // niter, # type: ignore
                                    false_fn=lambda: batch_size_test)
        batch_size_test: tf.Tensor = tf.cast(batch_size_test_tmp, dtype = tf.int32) # type: ignore
        tf.debugging.assert_positive(batch_size_test, message="batch_size_test should be positive integer and number of samples should be larger than number of iterations.")
        return niter, batch_size_test

    @property
    def param_dict(self) -> Dict[str, Any]:
        if self.Inputs.use_tf:
            niter, batch_size_test = self.get_niter_batch_size_np()
        else:
            niter, batch_size_test = self.get_niter_batch_size_tf()
        output_dict = self.Inputs.param_dict
        niter_used = niter
        output_dict["niter_used"] = int(niter_used) # type: ignore
        output_dict["batch_size_test_used"] = int(batch_size_test) # type: ignore
        output_dict["computing_time"] = self.get_computing_time()
        output_dict["small_sample_threshold"] = self.small_sample_threshold
        output_dict["small_sample"] = self.small_sample
        return output_dict    
            
    def get_computing_time(self) -> float:
        return self.end - self.start
    
    @abstractmethod
    def Test_np(self) -> None:
        pass
    
    @abstractmethod
    def Test_tf(self) -> None:
        pass
    

class TwoSampleTestSlicedBase(TwoSampleTestBase):
    """
    Base class for metrics.
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
        
        # Initialize base class
        super().__init__(data_input = data_input, 
                         progress_bar = progress_bar,
                         verbose = verbose)
    
        # From this class
        self._seed_slicing: int = seed_slicing if seed_slicing is not None else int(self.Inputs.seed_generator.make_seeds(1)[0,0]) # type: ignore
        self._nslices: int = nslices
        self._directions: DataTypeNP
        
        self.generate_directions()
        
    @property
    def seed_slicing(self) -> int:
        return self._seed_slicing
    
    @seed_slicing.setter
    def seed_slicing(self, seed_slicing: int) -> None:
        if isinstance(seed_slicing, int):
            self._seed_slicing = seed_slicing
            self.generate_directions()
        else:
            raise TypeError("seed_slicing must be an integer.")
        
    @property
    def nslices(self) -> int:
        return self._nslices
    
    @nslices.setter
    def nslices(self, nslices: int) -> None:
        if isinstance(nslices, int):
            self._nslices = nslices
            self.generate_directions()
        else:
            raise TypeError("nslices must be an integer.")
        
    @property
    def directions(self) -> DataTypeNP:
        return self._directions
            
    def generate_directions(self) -> None:
        """
        Function that generates random directions.
        Directions are always generated with the numpy backend for reproducibility
        """
        if self.verbose:
            print("Generating random directions based on nslices, ndims, and seed_slicing.")
        ndims: int = self.Inputs.ndims
        directions: DataTypeNP
        reset_random_seeds(seed = self.seed_slicing)
        directions = np.random.randn(self.nslices, self.Inputs.ndims)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        self._directions = directions