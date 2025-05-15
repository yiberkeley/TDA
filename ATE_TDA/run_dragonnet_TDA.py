"""
Main script to run DragonNet experiments with parallelization.
This file imports functions from dragonnet_module.py and should be run directly.
"""

import os
import numpy as np
import torch
import pandas as pd
import multiprocessing
from functools import partial

# Import all needed functions from dragonnet_module
from dragonnet_TDA import (
    gen_synthetic_data, 
    DragonNet,
    DragonTrainer,
    calculate_metrics,
    train_with_targeting_and_early_stopping,
    targeted_update_with_regularization,
    combined_training_and_post_targeting,
    train_with_tmle_style_loss,
    train_with_tmle_style_loss_and_update,
    run_experiment_with_tmle_approaches,
    summarize_tmle_approaches_results
)

def process_single_sim(rep, seed_offset=1000, is_ihdp=False, output_dir='simulation_results', **kwargs):
    """Process a single simulation replication"""
    seed_i = seed_offset + rep
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    
    # Determine if we're running IHDP or synthetic data
    try:
        if is_ihdp:
            # Load IHDP data
            try:
                train_data = np.load('datasets/ihdp_data/ihdp_npci_1-1000.train.npz')
                test_data = np.load('datasets/ihdp_data/ihdp_npci_1-1000.test.npz')
            except FileNotFoundError:
                print(f"Worker {rep}: IHDP dataset not found. Please download and place in datasets/ihdp_data/")
                return None
            
            x_train = train_data['x']
            x_train = np.transpose(x_train, (2, 0, 1))
            t_train = train_data['t']
            yf_train = train_data['yf']
            ycf_train = train_data['ycf']
            
            x_test = test_data['x']
            x_test = np.transpose(x_test, (2, 0, 1))
            t_test = test_data['t']
            yf_test = test_data['yf']
            ycf_test = test_data['ycf']
            
            # Concatenate training and testing data
            x_all = [np.concatenate([x_train[i], x_test[i]], axis=0) for i in range(1000)]
            t_all = np.concatenate([t_train, t_test], axis=0).transpose(1, 0)
            yf_all = np.concatenate([yf_train, yf_test], axis=0).transpose(1, 0)
            ycf_all = np.concatenate([ycf_train, ycf_test], axis=0).transpose(1, 0)
            
            df1 = pd.read_csv('datasets/ihdp_data/ihdp_treat_cov.csv')
            p_true = df1.loc[:, 'propscore'].to_numpy()
            
            Y = yf_all[rep]
            X = x_all[rep]
            A = t_all[rep]
            Y_cf = ycf_all[rep]
            
            tmp = pd.DataFrame({'Y': Y,'A': A,'Y_cf': Y_cf})
            true_ate_vec = tmp.apply(lambda d: d['Y'] - d['Y_cf'] if d['A']==1
                          else d['Y_cf'] - d['Y'], axis=1)
            true_ate = true_ate_vec.mean()
        else:
            # Generate synthetic data
            n = kwargs.get('n', 3000)
            X, A, Y, p_true = gen_synthetic_data(n=n, seed=seed_i)
            true_ate = 29.5  # Known for synthetic data
        
        # Run the experiment
        print(f"Worker {rep} (seed {seed_i}) starting experiment...")
        res_i = run_experiment_with_tmle_approaches(
            X, A, Y, p_true, true_ate, device='cpu',
            init_lambda=kwargs.get('init_lambda', 0.01),
            patience=kwargs.get('patience', 5),
            targeting_frequency=kwargs.get('targeting_frequency', 10),
            targeting_lambda=kwargs.get('targeting_lambda', 0.01),
            post_targeting_max_iter=kwargs.get('post_targeting_max_iter', 50),
            post_targeting_lambda=kwargs.get('post_targeting_lambda', 0.01),
            tmle_weight=kwargs.get('tmle_weight', 0.1),
            train_ratio=kwargs.get('train_ratio', 0.6),
            val_ratio=kwargs.get('val_ratio', 0.2),
            verbose=False
        )
        
        # Remove large metrics to save memory
        metrics_to_remove = [
            'metrics_post_last', 'metrics_post_full', 
            'metrics_ttr_last', 'metrics_ttr_full',
            'metrics_combined_last', 'metrics_combined_full',
            'metrics_tmle_loss', 'metrics_tmle_update'
        ]
        
        for key in metrics_to_remove:
            if key in res_i:
                res_i.pop(key, None)
        
        # Create result dictionary
        row_dict = dict(rep=rep, seed=seed_i)
        row_dict.update(res_i)
        
        # Save to temporary file
        os.makedirs(output_dir, exist_ok=True)
        temp_file = os.path.join(output_dir, f"temp_result_{rep}.csv")
        pd.DataFrame([row_dict]).to_csv(temp_file, index=False)
        
        print(f"Worker {rep} completed successfully")
        return temp_file
        
    except Exception as e:
        print(f"Worker {rep} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_parallel_simulations(n_sims, is_ihdp=False, **kwargs):
    """Run simulations in parallel"""
    # Setup output directory
    output_dir = "ihdp_tmle_approaches_results" if is_ihdp else "tmle_approaches_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Add output_dir to kwargs
    kwargs['output_dir'] = output_dir
    kwargs['is_ihdp'] = is_ihdp
    
    # Set number of processes
    n_processes = kwargs.pop('n_processes', None)
    if n_processes is None:
        n_processes = multiprocessing.cpu_count()
    n_processes = min(n_processes, n_sims)
    
    print(f"Running {n_sims} simulations with {n_processes} processes")
    
    # Create worker function with fixed params
    worker_fn = partial(process_single_sim, **kwargs)
    
    # If running with just 1 process, execute directly (easier for debugging)
    if n_processes == 1:
        result_files = [worker_fn(rep) for rep in range(n_sims)]
    else:
        # Run in parallel
        with multiprocessing.Pool(processes=n_processes) as pool:
            result_files = pool.map(worker_fn, range(n_sims))
    
    # Combine results
    final_results_path = os.path.join(output_dir, 
                                     "ihdp_tmle_approaches_results_summary.csv" if is_ihdp else
                                     "tmle_approaches_results_summary.csv")
    
    all_results = []
    for file_path in result_files:
        if file_path and os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                all_results.append(df)
                os.remove(file_path)  # Clean up temp file
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(final_results_path, index=False)
        print(f"Results saved to {final_results_path}")
        
        # Try to summarize results
        try:
            summary = summarize_tmle_approaches_results(final_results_path)
            if summary is not None:
                print("\n--- Results Summary ---")
                print(summary)
                summary.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
                return summary
        except Exception as e:
            print(f"Error summarizing results: {e}")
    else:
        print("No successful simulations to combine")
    
    return None

if __name__ == "__main__":
    # Ensure proper multiprocessing method
    multiprocessing.set_start_method('spawn')
    
    # Set device for entire script
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\n===== Running DragonNet with Parallelized TMLE-style Loss Approaches =====\n")
    
    # Run on synthetic data (easier to test with)
    run_parallel_simulations(
        n_sims=1000,  # Start small for testing
        is_ihdp=True,  # Use synthetic data
        seed_offset=2000,
        init_lambda=0.01,
        patience=10,  # Reduce for faster test runs
        targeting_frequency=10,
        targeting_lambda=0.01,
        post_targeting_max_iter=100,  # Reduce for faster test runs
        post_targeting_lambda=0.01,
        tmle_weight=0.1,
        train_ratio=0.8,
        val_ratio=0.2,
        n_processes=13  # Limit processes for testing
    )
    
    print("\n===== Parallelized TMLE Approaches Experiment Complete =====")