__all__ = ['compute_exclusion_bisection',
           'compute_exclusion_LR_bisection']
import os
import sys
from datetime import datetime
import numpy as np
import json
from scipy.stats import moment # type: ignore
import tensorflow as tf
import tensorflow_probability as tfp
from timeit import default_timer as timer

sys.path.insert(0,'../utils_func/')
sys.path.insert(0,'../')
import GMetrics # type: ignore
from GMetrics.utils import save_update_metrics_config, save_update_LR_metrics_config, se_mean, se_std # type: ignore
from . import MixtureDistributions # type: ignore

from typing import Tuple, Union, Optional, Type, Callable, Dict, List, Any
from numpy import typing as npt
DTypeType = Union[tf.DType, np.dtype, type]
IntTensor = Type[tf.Tensor]
FloatTensor = Type[tf.Tensor]
BoolTypeTF = Type[tf.Tensor]
BoolTypeNP = np.bool_
IntType = Union[int, IntTensor]
DataTypeTF = FloatTensor
DataTypeNP = npt.NDArray[np.float_]
DataType = Union[DataTypeNP, DataTypeTF]
DistTypeTF = tfp.distributions.Distribution

def compute_exclusion_bisection(reference_distribution: tfp.distributions.Distribution,
                                metric_config: Dict[str,Any],
                                test_kwargs: Dict[str,Any],
                                model_dir: str,
                                deformation: str = "mean", # could be 'mean', 'cov_diag', 'cov_off_diag', 'power_abs_up', 'power_abs_down', 'random_normal', 'random_uniform'
                                seed_dist: int = 0,
                                x_tol: float = 0.01,
                                fn_tol: float = 0.01,
                                bound: str = "central", # could be 'upper', 'central', 'lower'
                                eps_min: float = 0.,
                                eps_max: float = 1.,
                                max_iterations: int = 100,
                                save: bool = True,
                                filename: str = "exclusion_limits.json",
                                verbose: bool = True
                               ) -> Dict[str,Any]:
    if deformation not in ["mean", "cov_diag", "cov_off_diag", "power_abs_up", "power_abs_down", "random_normal", "random_uniform"]:
        raise ValueError(f"Invalid value for deformation: {deformation}")
    timestamp: str = datetime.now().isoformat()
    if verbose:
        print("\n======================================================")
        print(f"=============== {metric_config['name']} - {deformation} ===============")
        print("======================================================") 
        
    metric_config = dict(metric_config)
    test_kwargs = dict(test_kwargs)
    niter_null = test_kwargs.pop("niter_null")
    niter_alt = test_kwargs.pop("niter_alt")
    test_kwargs["niter"] = niter_alt
        
    metric_name = metric_config["name"]
    metric_class = eval(metric_config["class_name"])
    metric_kwargs = metric_config["kwargs"]
    metric_result_key = metric_config["result_key"]
    metric_scale_func = metric_config["scale_func"]
    max_vectorize = metric_config["max_vectorize"]
    
    # Define ndims
    ndims = metric_config["test_config"]["ndims"]
    
    # Compute metric scaling factor
    nsamples = test_kwargs["batch_size_test"]
    ns = nsamples**2 / (2 * nsamples)
    
    dist_1 = reference_distribution
    start_global = timer()

    metrics_list = []
    metrics_mean_list = []
    metrics_std_list = []
    eps_list = []
    exclusion_list = [["CL", "metric_name", "bound", "epsilon_value", "metric_tested", "metric_mean", "metric_std", "test_timing"]]

    metric_thresholds = metric_config["thresholds"][-2:]
    metric_threshold_number = 0
    eps_min_start = eps_min
    eps_max_start = eps_max
    eps = (eps_max + eps_min) / 2.
    eps_min, eps_max = eps_min_start, eps_max_start # Initialize the bounds

    start = timer()
    
    iteration = 0

    while metric_threshold_number < len(metric_thresholds) and iteration < max_iterations:
        iteration += 1
        
        if deformation in ["mean", "cov_diag", "cov_off_diag"]:
            deform_kwargs = {"eps": eps, "deform_type": deformation, "seed": seed_dist}
        elif "power_abs" in deformation:
            deform_type = "power_abs"
            direction = deformation.split("_")[-1]
            deform_kwargs = {"eps": eps, "deform_type": deform_type, "direction": direction}
        elif "random" in deformation:
            deform_type = "random"
            shift_dist = deformation.split("_")[-1]
            deform_kwargs = {"eps": eps, "deform_type": deform_type, "shift_dist": shift_dist, "seed": seed_dist}
        else:
            raise ValueError(f"Invalid value for deformation: {deformation}")
        
        print(f"\n------------ {iteration} - {metric_thresholds[metric_threshold_number][0]} CL - {bound} bound -----------")
        print(f"eps = {eps} - deformation = {deformation}")
        
        dist_2 = MixtureDistributions.deformed_distribution(dist_1,
                                                            **deform_kwargs)
    
        TwoSampleTestInputs = GMetrics.TwoSampleTestInputs(dist_1_input = dist_1,
                                                           dist_2_input = dist_2,
                                                           **test_kwargs)
        
        Metric = metric_class(TwoSampleTestInputs, **metric_kwargs) # type: ignore
        Metric.Test_tf(max_vectorize=max_vectorize)
        metric_mean = np.mean(Metric.Results[-1].result_value[metric_result_key] * metric_scale_func(ns, ndims)) # type: ignore
        metric_std = np.std(Metric.Results[-1].result_value[metric_result_key] * metric_scale_func(ns, ndims)) # type: ignore
        
        if bound == "upper":
            metric = metric_mean - metric_std
        elif bound == "central":
            metric = metric_mean
        elif bound == "lower":
            metric = metric_mean + metric_std

        metrics_list.append(metric)
        metrics_mean_list.append(metric_mean)
        metrics_std_list.append(metric_std)
        eps_list.append(eps)
            
        # Determine direction of adjustment based on overshooting or undershooting
        if metric > metric_thresholds[metric_threshold_number][2]: # type: ignore
            #direction = -1
            eps_max = eps  # Update the maximum bound
            eps = eps_max - (eps_max - eps_min) / 2.
        else:
            #direction = 1
            eps_min = eps  # Update the minimum bound
            eps = eps_min + (eps_max - eps_min) / 2.

        if verbose:
            print(f"statistic = {metric} - next threshold = {metric_thresholds[metric_threshold_number][2]} at {metric_thresholds[metric_threshold_number][0]} CL")

        relative_error_eps = 2 * (eps_max - eps_min) / (eps_max + eps_min)
        relative_error_metric = 2 * np.abs(metric_thresholds[metric_threshold_number][2] - metric) / (metric_thresholds[metric_threshold_number][2] + metric)
        if verbose:
            print(f"relative_error_eps = {relative_error_eps}")
            print(f"relative_error_metric = {relative_error_metric}")

        # Check if the fn value is within the required accuracy of the threshold
        if relative_error_eps < x_tol and relative_error_metric < fn_tol:
            end = timer()
            if verbose:
                print(f"=======> statistic within required accuracy at {metric_thresholds[metric_threshold_number][0]} CL in {end - start} seconds")
            exclusion_list.append([metric_thresholds[metric_threshold_number][0], metric_name, bound, eps, metric, metric_mean, metric_std, end - start]) # type: ignore
            metric_threshold_number += 1
            print("\n======================================================")
            print("New threshold. Resetting eps_min and eps_max.")
            start = timer() # Reset the timer
            iteration = 0
            eps_min, eps_max = eps, eps_max_start # Initialize the bounds
            
        if iteration == max_iterations - 1:
            end = timer()
            exclusion_list.append([metric_thresholds[metric_threshold_number][0], metric_name, bound, None, None, None, None, end - start]) # type: ignore
        
    end = timer()
    if verbose:
        print("Time elapsed:", end - start_global, "seconds.")
    result = {metric_name+"_"+deformation+"_"+bound+"_"+timestamp: {"name": metric_config["name"],
                                                                    "deformation": deformation,
                                                                    "bound": bound,
                                                                    "parameters": {"seed_dist": seed_dist,
                                                                                   "x_tol": x_tol,
                                                                                   "fn_tol": fn_tol,
                                                                                   "eps_min": eps_min_start,
                                                                                   "eps_max": eps_max_start,
                                                                                   "max_iterations": max_iterations,
                                                                                   "save": save,
                                                                                   "filename": filename,
                                                                                   "verbose": verbose},
                                                                    "exclusion_list": exclusion_list,
                                                                    "eps_list": eps_list,
                                                                    "metrics_list": metrics_list,
                                                                    "metrics_mean_list": metrics_mean_list,
                                                                    "metrics_std_list": metrics_std_list,
                                                                    "time_elapsed": end - start_global,
                                                                    "test_config": test_kwargs,
                                                                    "null_config": metric_config}}
    
    # Saving if required
    if save:
        file_path = os.path.join(model_dir,filename)
        if verbose:
            print(f"Saving results in the file {file_path}")
        # Step 1: Read the existing content if the file exists
        existing_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as file:
                    existing_data = json.load(file)
            except json.JSONDecodeError:
                # File is empty or corrupted, start with an empty dictionary
                existing_data = {}
        
        # Step 2: Update the dictionary with new results
        existing_data.update(result)
        
        # Step 3: Write the updated dictionary back to the file
        # Use this custom encoder when dumping your JSON data
        with open(file_path, "w") as file:
            json.dump(existing_data, file, cls=GMetrics.utils.CustomEncoder, indent=4) # type: ignore
    
    return result

def compute_exclusion_LR_bisection(reference_distribution: tfp.distributions.Distribution,
                                   metric_config: Dict[str,Any],
                                   test_kwargs: Dict[str,Any],
                                   model_dir: str,
                                   cl_list = [0.95, 0.99],
                                   deformation: str = "mean", # could be 'mean', 'cov_diag', 'cov_off_diag', 'power_abs_up', 'power_abs_down', 'random_normal', 'random_uniform'
                                   seed_dist: int = 0,
                                   x_tol: float = 0.01,
                                   fn_tol: float = 0.01,
                                   bound: str = "central", # could be 'upper', 'central', 'lower'
                                   eps_min: float = 0.,
                                   eps_max: float = 1.,
                                   max_iterations: int = 100,
                                   save: bool = True,
                                   filename: str = "exclusion_limits.json",
                                   verbose: bool = True
                                  ) -> Dict[str,Any]:
    if deformation not in ["mean", "cov_diag", "cov_off_diag", "power_abs_up", "power_abs_down", "random_normal", "random_uniform"]:
        raise ValueError(f"Invalid value for deformation: {deformation}")
    timestamp: str = datetime.now().isoformat()
    if verbose:
        print("\n======================================================")
        print(f"=============== {metric_config['name']} - {deformation} ===============")
        print("======================================================") 
        
    metric_config = dict(metric_config)
    test_kwargs = dict(test_kwargs)
    niter_null = test_kwargs.pop("niter_null")
    niter_alt = test_kwargs.pop("niter_alt")
    
    test_kwargs_null = dict(test_kwargs)
    test_kwargs_null["niter"] = niter_null
    test_kwargs_alt = dict(test_kwargs)
    test_kwargs_alt["niter"] = niter_alt
    
    metric_kwargs_null = dict(metric_config["kwargs"])
    metric_kwargs_alt = dict(metric_config["kwargs"])
    metric_kwargs_null["null_test"] = True
    metric_kwargs_alt["null_test"] = False
    #metric_kwargs_null["verbose"] = True
    #metric_kwargs_alt["verbose"] = True
    
    metric_name = metric_config["name"]
    metric_result_key = metric_config["result_key"]
    metric_scale_func = metric_config["scale_func"]
    max_vectorize = metric_config["max_vectorize"]
    null_file_base = metric_config["null_file"]
    metrics_config_file = model_dir + "metrics_config.json"
    
    # Define ncomp and ndims
    ncomp = metric_config["test_config"]["ncomp"]
    ndims = metric_config["test_config"]["ndims"]
    
    # Compute metric scaling factor
    nsamples = test_kwargs["batch_size_test"]
    ns = nsamples**2 / (2 * nsamples)
    
    metrics_list = []
    metrics_mean_list = []
    metrics_std_list = []
    eps_list = []
    exclusion_list = []
    
    metric_threshold_number = 0
    eps_min_start = eps_min
    eps_max_start = eps_max
    eps = (eps_max + eps_min) / 2.
    
    start_global = timer()
    start = timer()
    
    dist_1 = reference_distribution
    
    iteration = 0

    while metric_threshold_number < len(cl_list) and iteration < max_iterations:
        iteration += 1
        
        if deformation in ["mean", "cov_diag", "cov_off_diag"]:
            deform_kwargs = {"eps": eps, "deform_type": deformation, "seed": seed_dist}
        elif "power_abs" in deformation:
            deform_type = "power_abs"
            direction = deformation.split("_")[-1]
            deform_kwargs = {"eps": eps, "deform_type": deform_type, "direction": direction}
        elif "random" in deformation:
            deform_type = "random"
            shift_dist = deformation.split("_")[-1]
            deform_kwargs = {"eps": eps, "deform_type": deform_type, "shift_dist": shift_dist, "seed": seed_dist}
        else:
            raise ValueError(f"Invalid value for deformation: {deformation}")
            
        print(f"\n------------ {iteration} - {cl_list[metric_threshold_number]} CL - {bound} bound -----------")
        print(f"eps = {eps} - deformation = {deformation}")

        print(f"Computing null distribution")
        start_null = timer()
        dist_2 = MixtureDistributions.deformed_distribution(dist_1,
                                                            **deform_kwargs)
        
        TwoSampleTestInputs = GMetrics.TwoSampleTestInputs(dist_1_input = dist_1,
                                                           dist_2_input = dist_2,
                                                           **test_kwargs_null)
        
        LRMetric_null = GMetrics.LRMetric(TwoSampleTestInputs, **metric_kwargs_null)
        LRMetric_null.Test_tf(max_vectorize = max_vectorize)
        
        null_file = null_file_base.replace(".json", "_" + deformation + "_" + str(format(eps, '.6f')) + ".json")
        print("Saving", metric_name, "to", null_file)
        LRMetric_null.Results.save_to_json(null_file)
        
        dist_null = np.array(LRMetric_null.Results[-1].result_value[metric_result_key]) * metric_scale_func(ns, ndims)
        metric_thresholds = [[cl, 
                              [int(cl*len(dist_null)), 
                               int((1-cl)*len(dist_null))], 
                              np.sort(dist_null)[int(len(dist_null)*cl)]] for cl in cl_list]
        print(f"ThresholdS: {metric_thresholds}")
        metric_config["thresholds"].append([eps, deformation, metric_thresholds])
        save_update_LR_metrics_config(metric_config = metric_config, 
                                      metrics_config_file = metrics_config_file) # type: ignore
        end_null = timer()
        print(f"Null distribution computed in {end_null - start_null} seconds")
        
        print(f"Evaluating alternative distribution")
        TwoSampleTestInputs = GMetrics.TwoSampleTestInputs(dist_1_input = dist_1,
                                                           dist_2_input = dist_2,
                                                           **test_kwargs_alt)
        LRMetric_alt = GMetrics.LRMetric(TwoSampleTestInputs, **metric_kwargs_alt)
        LRMetric_alt.Test_tf(max_vectorize = max_vectorize)
        metric_mean = np.mean(LRMetric_alt.Results[-1].result_value[metric_result_key]) * metric_scale_func(ns, ndims) # type: ignore
        metric_std = np.mean(LRMetric_alt.Results[-1].result_value[metric_result_key]) * metric_scale_func(ns, ndims) # type: ignore
        
        if bound == "upper":
            metric = metric_mean - metric_std
        elif bound == "central":
            metric = metric_mean
        elif bound == "lower":
            metric = metric_mean + metric_std

        metrics_list.append(metric)
        metrics_mean_list.append(metric_mean)
        metrics_std_list.append(metric_std)
        eps_list.append(eps)

        # Determine direction of adjustment based on overshooting or undershooting
        if metric > metric_thresholds[metric_threshold_number][2]: # type: ignore
            #direction = -1
            eps_max = eps  # Update the maximum bound
            eps = eps_max - (eps_max - eps_min) / 2.
        else:
            #direction = 1
            eps_min = eps  # Update the minimum bound
            eps = eps_min + (eps_max - eps_min) / 2.
                        
        if verbose:
            print(f"statistic = {metric} - next threshold = {metric_thresholds[metric_threshold_number][2]} at {metric_thresholds[metric_threshold_number][0]} CL")

        relative_error_eps = 2 * (eps_max - eps_min) / (eps_max + eps_min)
        relative_error_metric = 2 * np.abs(metric_thresholds[metric_threshold_number][2] - metric) / (metric_thresholds[metric_threshold_number][2] + metric)
        if verbose:
            print(f"relative_error_eps = {relative_error_eps}")
            print(f"relative_error_metric = {relative_error_metric}")
         
        # Check if the fn value is within the required accuracy of the threshold
        if relative_error_eps < x_tol and relative_error_metric < fn_tol:
            end = timer()
            if verbose:
                print(f"=======> statistic within required accuracy at {metric_thresholds[metric_threshold_number][0]} CL in {end - start} seconds")
            exclusion_list.append([metric_thresholds[metric_threshold_number][0], metric_name, bound, eps, metric, metric_mean, metric_std, end - start]) # type: ignore
            metric_threshold_number += 1
            print("\n======================================================")
            print("New threshold. Resetting eps_min and eps_max.")
            start = timer() # Reset the timer
            iteration = 0
            eps_min, eps_max = eps, eps_max_start # Initialize the bounds
            
        if iteration == max_iterations - 1:
            end = timer()
            exclusion_list.append([metric_thresholds[metric_threshold_number][0], metric_name, bound, None, None, None, None, end - start]) # type: ignore
            
    end = timer()
    if verbose:
        print("Time elapsed:", end - start_global, "seconds.")
    result = {metric_name+"_"+deformation+"_"+bound+"_"+timestamp: {"name": metric_config["name"],
                                                                    "deformation": deformation,
                                                                    "bound": bound,
                                                                    "parameters": {"seed_dist": seed_dist,
                                                                                   "x_tol": x_tol,
                                                                                   "fn_tol": fn_tol,
                                                                                   "eps_min": eps_min_start,
                                                                                   "eps_max": eps_max_start,
                                                                                   "max_iterations": max_iterations,
                                                                                   "save": save,
                                                                                   "filename": filename,
                                                                                   "verbose": verbose},
                                                                    "exclusion_list": exclusion_list,
                                                                    "eps_list": eps_list,
                                                                    "metrics_list": metrics_list,
                                                                    "metrics_mean_list": metrics_mean_list,
                                                                    "metrics_std_list": metrics_std_list,
                                                                    "time_elapsed": end - start_global,
                                                                    "test_config": test_kwargs,
                                                                    "null_config": metric_config}}
    # Saving if required
    if save:
        file_path = os.path.join(model_dir,filename)
        if verbose:
            print(f"Saving results in the file {file_path}")
        # Step 1: Read the existing content if the file exists
        existing_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as file:
                    existing_data = json.load(file)
            except json.JSONDecodeError:
                # File is empty or corrupted, start with an empty dictionary
                existing_data = {}
        
        # Step 2: Update the dictionary with new results
        existing_data.update(result)
        
        # Step 3: Write the updated dictionary back to the file
        # Use this custom encoder when dumping your JSON data
        with open(file_path, "w") as file:
            json.dump(existing_data, file, cls=GMetrics.utils.CustomEncoder, indent=4) # type: ignore
    
    return result