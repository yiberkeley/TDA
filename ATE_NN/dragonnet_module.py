"""
DragonNet module - Contains all functions and classes for the DragonNet model.
This module is designed to be imported by other scripts.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import os

###############################################################################
# 1) DATA GENERATION
###############################################################################
def gen_synthetic_data(n=3000, seed=123):
    """
    Generates synthetic data with a known ATE ~ 29.5.

    Returns:
        X: (n,1) numpy array of features.
        A: (n,) numpy array of binary treatments.
        Y: (n,) numpy array of outcomes.
        p_true: the true propensity scores.
    """
    np.random.seed(seed)
    X = np.random.uniform(0, 1, size=n)
    z = -0.02 * X - X**2 + 4 * np.log(X + 0.3) + 1
    p_x = 1.0 / (1.0 + np.exp(-z))
    A = np.random.binomial(1, p_x)
    mu = 5 * X + 9 * A * X + 5 * np.sin(np.pi * X) + 25 * (A - 2)
    Y = mu + np.random.randn(n)
    return X.reshape(-1, 1), A.astype(float), Y, p_x

###############################################################################
# 2) DRAGONNET MODEL DEFINITION
###############################################################################
class DragonNet(nn.Module):
    """
    DragonNet with a common trunk, a propensity head and two outcome heads (u0, u1).
    Also contains a trainable ATE scalar parameter (which we do not update in projection).
    """
    def __init__(self, xdim=1, hidden_shared=200, hidden_outcome=100):
        super().__init__()
        # Shared trunk
        self.trunk_l1 = nn.Linear(xdim, hidden_shared)
        self.trunk_a1 = nn.ELU()
        self.trunk_l2 = nn.Linear(hidden_shared, hidden_shared)
        self.trunk_a2 = nn.ELU()
        # Propensity head
        self.prop_l1 = nn.Linear(hidden_shared, hidden_shared)
        self.prop_a1 = nn.ELU()
        self.prop_out = nn.Linear(hidden_shared, 1)  # apply sigmoid in forward
        # Outcome head for control (u0)
        self.mu0_l1 = nn.Linear(hidden_shared, hidden_outcome)
        self.mu0_a1 = nn.ELU()
        self.mu0_out = nn.Linear(hidden_outcome, 1)
        # Outcome head for treatment (u1)
        self.mu1_l1 = nn.Linear(hidden_shared, hidden_outcome)
        self.mu1_a1 = nn.ELU()
        self.mu1_out = nn.Linear(hidden_outcome, 1)
        # ATE parameter (not updated in projection)
        self.ate = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, x):
        # Shared trunk processing
        h = self.trunk_a1(self.trunk_l1(x))
        h = self.trunk_a2(self.trunk_l2(h))
        # Propensity prediction (apply sigmoid)
        hp = self.prop_a1(self.prop_l1(h))
        p_hat = torch.sigmoid(self.prop_out(hp))
        # Outcome predictions
        h0 = self.mu0_a1(self.mu0_l1(h))
        mu0 = self.mu0_out(h0)
        h1 = self.mu1_a1(self.mu1_l1(h))
        mu1 = self.mu1_out(h1)
        return p_hat, mu0, mu1, self.ate

###############################################################################
# 3) STANDARD TRAINER (BCE + MSE LOSS)
###############################################################################
class DragonTrainer:
    def __init__(self, net, device='cpu'):
        self.net = net.to(device)
        self.device = device

    def train(self, X, A, Y, n_epochs=200, batch_size=128, lr=1e-3,
              weight_decay=1e-5, verbose=True):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        A_t = torch.tensor(A, dtype=torch.float32).view(-1, 1).to(self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32).view(-1, 1).to(self.device)
        ds = torch.utils.data.TensorDataset(X_t, A_t, Y_t)
        if batch_size is None or batch_size >= len(X):
            loader = [(X_t, A_t, Y_t)]
        else:
            loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        # Do not update the ATE parameter here.
        params_std = [p for n, p in self.net.named_parameters() if n != 'ate']
        opt = optim.Adam(params_std, lr=lr, weight_decay=weight_decay)
        for ep in range(n_epochs):
            self.net.train()
            for (xb, ab, yb) in loader:
                p_hat, mu0, mu1, _ = self.net(xb)
                bce_loss = F.binary_cross_entropy(p_hat, ab)
                mask0 = (ab < 0.5)
                mask1 = (ab > 0.5)
                mse0 = torch.mean((yb[mask0] - mu0[mask0])**2) if mask0.any() else 0.0
                mse1 = torch.mean((yb[mask1] - mu1[mask1])**2) if mask1.any() else 0.0
                loss = bce_loss + mse0 + mse1
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params_std, 5.0)
                opt.step()
            if verbose and ep % 50 == 0:
                print(f"[Stage1 ep={ep}] loss={loss.item():.4f}")
    
    def train_epoch(self, X, A, Y, batch_size=128, lr=1e-3, weight_decay=1e-5):
        """Train for a single epoch and return the loss"""
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        A_t = torch.tensor(A, dtype=torch.float32).view(-1, 1).to(self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32).view(-1, 1).to(self.device)
        ds = torch.utils.data.TensorDataset(X_t, A_t, Y_t)
        if batch_size is None or batch_size >= len(X):
            loader = [(X_t, A_t, Y_t)]
        else:
            loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        
        # Do not update the ATE parameter here
        params_std = [p for n, p in self.net.named_parameters() if n != 'ate']
        opt = optim.Adam(params_std, lr=lr, weight_decay=weight_decay)
        
        self.net.train()
        total_loss = 0.0
        num_batches = 0
        
        for (xb, ab, yb) in loader:
            p_hat, mu0, mu1, _ = self.net(xb)
            bce_loss = F.binary_cross_entropy(p_hat, ab)
            mask0 = (ab < 0.5)
            mask1 = (ab > 0.5)
            mse0 = torch.mean((yb[mask0] - mu0[mask0])**2) if mask0.any() else 0.0
            mse1 = torch.mean((yb[mask1] - mu1[mask1])**2) if mask1.any() else 0.0
            loss = bce_loss + mse0 + mse1
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_std, 5.0)
            opt.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0

    def evaluate(self, X, A, Y):
        """Evaluate the model on given data and return the loss"""
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        A_t = torch.tensor(A, dtype=torch.float32).view(-1, 1).to(self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32).view(-1, 1).to(self.device)
        
        self.net.eval()
        with torch.no_grad():
            p_hat, mu0, mu1, _ = self.net(X_t)
            bce_loss = F.binary_cross_entropy(p_hat, A_t)
            mask0 = (A_t < 0.5)
            mask1 = (A_t > 0.5)
            mse0 = torch.mean((Y_t[mask0] - mu0[mask0])**2) if mask0.any() else 0.0
            mse1 = torch.mean((Y_t[mask1] - mu1[mask1])**2) if mask1.any() else 0.0
            loss = bce_loss + mse0 + mse1
        
        return loss.item()

    def predict(self, X):
        self.net.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            p_hat, mu0, mu1, _ = self.net(X_t)
        return (p_hat.cpu().numpy().ravel(),
                mu0.cpu().numpy().ravel(),
                mu1.cpu().numpy().ravel())

###############################################################################
# 4) HELPER: Compute delta_i for one sample (outcome residual)
###############################################################################
def delta_i_for_sample(net, x_i, a_i, y_i, p_i):
    """
    Computes the outcome residual:
      delta_i = (A / p) * (Y - u1) - ((1-A) / (1-p)) * (Y - u0)
    """
    _, mu0, mu1, _ = net(x_i)
    mu0_ = mu0.squeeze()
    mu1_ = mu1.squeeze()
    delta = (a_i / (p_i + 1e-8)) * (y_i - mu1_) - ((1.0 - a_i) / ((1.0 - p_i) + 1e-8)) * (y_i - mu0_)
    return delta

###############################################################################
# 4.1) HELPER: Compute MSE for one sample (for gradient computation)
###############################################################################
def mse_loss_for_sample(net, x_i, a_i, y_i):
    """
    Computes the MSE loss for a single sample based on treatment status.
    """
    _, mu0, mu1, _ = net(x_i)
    mu0_ = mu0.squeeze()
    mu1_ = mu1.squeeze()

    # Compute MSE based on treatment status
    if a_i > 0.5:  # Treated sample
        loss = (y_i - mu1_)**2
    else:  # Control sample
        loss = (y_i - mu0_)**2

    return loss

###############################################################################
# 5) BUILD HYBRID PROJECTION SYSTEM (MSE GRADIENTS BUT DELTA_I VALUES)
###############################################################################
def build_hybrid_projection_system(net, X_data, A_data, Y_data, p_data, param_list, device='cpu'):
    """
    For each data sample, compute:
      - r: the delta_i residual (keeping original r vector)
      - G: the gradient of -MSE loss (new way to compute G)

    Returns:
      r_vec: (n_data,) tensor of delta_i values.
      G: (n_data, param_dim) tensor stacking gradients of MSE.
    """
    n_data = len(X_data)
    X_t = torch.tensor(X_data, dtype=torch.float32).to(device)
    A_t = torch.tensor(A_data, dtype=torch.float32).view(-1, 1).to(device)
    Y_t = torch.tensor(Y_data, dtype=torch.float32).view(-1, 1).to(device)
    p_t = torch.tensor(p_data, dtype=torch.float32).view(-1, 1).to(device)

    r_list = []  # Will contain delta_i values
    grad_list = []  # Will contain MSE gradients

    for i in range(n_data):
        x_i = X_t[i:i+1, :]
        a_i = A_t[i].item()
        y_i = Y_t[i].item()
        p_i = p_t[i].item()

        # Do a truncation for positivity control ##NOTE
        p_i = np.clip(p_i, 5 / (np.sqrt(n_data) * np.log(n_data)),1- 5 / (np.sqrt(n_data) * np.log(n_data)))

        # First, compute delta_i (original r values)
        with torch.no_grad():  # No need for gradients when computing delta_i
            delta_val = delta_i_for_sample(net, x_i, a_i, y_i, p_i)
            r_list.append(delta_val.item())

        # Make sure all parameters have requires_grad=True before computing MSE gradients
        for param in param_list:
            param.requires_grad = True

        # Now, compute gradients of MSE loss (new G values)
        net.zero_grad()
        mse_val = mse_loss_for_sample(net, x_i, a_i, y_i)
        mse_val.backward(retain_graph=True)

        # Gather gradients from parameters in param_list
        grads = []
        for p in param_list:
            # Check if grad is None and handle it
            if p.grad is None:
                # If gradient is None, use zeros instead
                grads.append(torch.zeros_like(p).flatten())
            else:
                grads.append(p.grad.clone().flatten())
        grad_vec = torch.cat(grads, dim=0)
        grad_list.append(grad_vec)
        net.zero_grad()

    r_vec = torch.tensor(r_list, dtype=torch.float32, device=device)
    G = torch.stack(grad_list)
    G = -G  # Use negative MSE gradients
    return r_vec, G

###############################################################################
# 6) CALCULATE MSE HELPER FUNCTION
###############################################################################
def calculate_mse(net, X, A, Y, device='cpu'):
    """
    Calculate MSE for a network on a dataset
    """
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    A_t = torch.tensor(A, dtype=torch.float32).to(device)
    Y_t = torch.tensor(Y, dtype=torch.float32).to(device)

    with torch.no_grad():
        p_hat, mu0, mu1, _ = net(X_t)
        mu0 = mu0.squeeze().cpu().numpy()
        mu1 = mu1.squeeze().cpu().numpy()
        A_np = A_t.cpu().numpy()
        Y_np = Y_t.cpu().numpy()

        # Calculate MSE based on treatment status
        mse = np.mean([(Y_np[i] - mu1[i])**2 if A_np[i] > 0.5 else (Y_np[i] - mu0[i])**2 for i in range(len(Y_np))])

    return mse

###############################################################################
# 7) PROJECTION STEP WITH REGULARIZED LEAST SQUARES
###############################################################################
def projection_step_with_regularization(net, X_data, A_data, Y_data, p_data, param_list,
                                        lambda_reg=0.01, device='cpu', prune_threshold=1e-3, verbose = False):
    """
    Performs a projection step using regularized least squares:
      1) Build (r, G) where r are the delta_i values and G is the -MSE gradient matrix.
      2) Solve the regularized least squares problem (G^T G + λI) Δw = G^T r.
      3) Set to zero any entries in Δw whose absolute values fall below prune_threshold.
      4) Find optimal epsilon through line search (including both positive and negative values).
      5) Update each parameter in param_list: w <- w + epsilon_optimal * (Δw segment).
      6) Return the vector r (for monitoring) and the selected epsilon value.
    """
    # Use the hybrid projection system
    r_vec, G = build_hybrid_projection_system(net, X_data, A_data, Y_data, p_data, param_list, device=device)

    # Check for all-zero gradients
    if torch.all(G == 0):
        print("Warning: All gradients are zero. Skipping this projection step.")
        return r_vec.cpu().numpy(), 0.0

    # REGULARIZED LEAST SQUARES SOLUTION
    # Add a column of ones to G for the intercept term
    ones_column = torch.ones((G.shape[0], 1), device=device)
    G_with_intercept = torch.cat([ones_column, G], dim=1)

    # Construct the normal equations with regularization
    # Note: We add regularization only to the coefficient terms, not to the intercept
    G_T = G_with_intercept.t()
    GTG = G_T @ G_with_intercept

    # Create regularization matrix (identity matrix with zero at the intercept position)
    n = GTG.shape[0]
    reg_matrix = torch.eye(n, device=device)
    reg_matrix[0, 0] = 0.0  # Don't regularize the intercept term

    # Apply regularization
    regularized_GTG = GTG + lambda_reg * reg_matrix

    # Solve the regularized normal equations
    rhs = G_T @ r_vec

    try:
        solution = torch.linalg.solve(regularized_GTG, rhs)
        if verbose:
            print(f"Using direct solve with lambda={lambda_reg:.6f}")
    except:
        # Fallback to SVD-based least squares if direct solve fails
        print(f"Direct solve failed. Using SVD with lambda={lambda_reg:.6f}")
        # Add a small value to diagonal for numerical stability
        regularized_GTG.diagonal().add_(1e-10)
        solution = torch.linalg.lstsq(regularized_GTG, rhs).solution

    # Separate the intercept and the coefficient vector
    intercept = solution[0].item()
    Delta_w = solution[1:]  # Discard the intercept, only keep coefficients

    if verbose:
        print(f"Fitted intercept (not used for updates): {intercept:.6f}")

    # Prune small coefficients
    Delta_w = torch.where(torch.abs(Delta_w) < prune_threshold, torch.tensor(0.0, device=device), Delta_w)

    # Print summary of Delta_w
    n_nonzero = torch.sum(Delta_w != 0).item()
    if n_nonzero > 0:
        avg_mag = torch.mean(torch.abs(Delta_w[Delta_w != 0])).item()
        if verbose:
            print(f"Delta_w: {n_nonzero} non-zero elements with avg magnitude {avg_mag:.6f}")
    else:
        if verbose:
            print("Delta_w: All elements are zero (or close to zero)")

    # Store parameter name mapping for the deep copy
    param_names = {}
    for name, param in net.named_parameters():
        for p in param_list:
            if param is p:
                param_names[id(p)] = name

    # Define a range of epsilon values to try, including negative values
    epsilon_values = [-1.0, -0.5, -0.1, -0.05, -0.01, -0.005, -0.001, -0.0005, -0.0001,
                      0.0,
                      0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    best_epsilon = 0.0
    min_mse = float('inf')

    # Find the best epsilon through line search (without excessive printing)
    for eps in epsilon_values:
        # Create a temporary copy of the network with updated parameters
        temp_net = deepcopy(net)

        # Apply the update with current epsilon
        idx = 0
        for p_orig in param_list:
            numel_p = p_orig.numel()
            delta_p = Delta_w[idx:idx+numel_p].view_as(p_orig)

            # Find the corresponding parameter in temp_net
            p_name = param_names[id(p_orig)]
            for name, p_temp in temp_net.named_parameters():
                if name == p_name:
                    with torch.no_grad():
                        p_temp.copy_(p_orig + eps * delta_p)
                    break

            idx += numel_p

        # Evaluate MSE on the data set
        mse = calculate_mse(temp_net, X_data, A_data, Y_data, device)

        # No printing of individual line search steps
        if mse < min_mse:
            min_mse = mse
            best_epsilon = eps

    #print(f"Selected epsilon: {best_epsilon:+.6f}")

    # Apply the update with the optimal epsilon
    idx = 0
    for p in param_list:
        numel_p = p.numel()
        delta_p = Delta_w[idx:idx+numel_p].view_as(p)
        with torch.no_grad():
            p.add_(best_epsilon * delta_p)
        idx += numel_p

    # Re-evaluate the EIC
    r_vec, G = build_hybrid_projection_system(net, X_data, A_data, Y_data, p_data, param_list, device=device)

    return r_vec.cpu().numpy(), best_epsilon

###############################################################################
# 8) HELPER: Calculate metrics (MSE and mean(u1-u0)) on a dataset
###############################################################################
def calculate_metrics(net, X, A, Y, device='cpu'):
    """
    Calculate MSE and mean(u1-u0) on a dataset
    """
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    A_t = torch.tensor(A, dtype=torch.float32).to(device)
    Y_t = torch.tensor(Y, dtype=torch.float32).to(device)

    with torch.no_grad():
        p_hat, mu0, mu1, _ = net(X_t)
        mu0 = mu0.squeeze().cpu().numpy()
        mu1 = mu1.squeeze().cpu().numpy()

        # Calculate mean(u1-u0)
        mean_diff = np.mean(mu1 - mu0)

        # Calculate MSE
        mse = np.mean([(Y[i] - mu1[i])**2 if A[i] > 0.5 else (Y[i] - mu0[i])**2 for i in range(len(Y))])

    return mse, mean_diff, mu1, mu0

###############################################################################
# 9) TARGETED UPDATE WITH REGULARIZATION AND ADAPTIVE STRATEGIES
###############################################################################
def targeted_update_with_regularization(net, X_data, A_data, Y_data, p_data,
                                       max_iter=50, device='cpu', prune_threshold=0,
                                       verbose=False, patience=5, init_lambda=0.01, layers_full=False):
    """
    Iteratively update the last layer weights of outcome heads using regularized least squares
    with adaptive regularization strength and early stopping.
    """
    net = net.to(device)
    net.eval()

    # Select the last TWO layers of parameters from outcome heads to update
    if(layers_full):
        outcome_names = [
        'mu0_l1.weight', # 'mu0_l1.bias',   # first layer of outcome head for u0
        'mu0_out.weight',# 'mu0_out.bias', # second (last) layer for u0
        'mu1_l1.weight', #'mu1_l1.bias',   # first layer of outcome head for u1
        'mu1_out.weight' # 'mu1_out.bias'  # second (last) layer for u1
        ]
    else:
        outcome_names = [
            #'mu0_l1.weight', # 'mu0_l1.bias',   # first layer of outcome head for u0
            'mu0_out.weight',# 'mu0_out.bias', # second (last) layer for u0
            #'mu1_l1.weight', #'mu1_l1.bias',   # first layer of outcome head for u1
            'mu1_out.weight' # 'mu1_out.bias'  # second (last) layer for u1
        ]

    params_targ = []
    for nm, param in net.named_parameters():
        if nm in outcome_names:
            param.requires_grad = True
            params_targ.append(param)
        else:
            param.requires_grad = False

    n_data = len(X_data)

    # Lists to track metrics
    mse_values = []          # Data MSE
    ate_values = []          # Model-based ATE
    mean_r_values = []       # Mean of delta_i
    epsilon_values = []      # Selected epsilon values
    lambda_values = []       # Regularization strength values

    # Initial metrics
    initial_mse, initial_ate,_,_ = calculate_metrics(net, X_data, A_data, Y_data, device)

    mse_values.append(initial_mse)
    ate_values.append(initial_ate)
    mean_r_values.append(0.0)  # Placeholder for initial mean(r)
    epsilon_values.append(0.0)  # Placeholder for initial epsilon
    lambda_values.append(init_lambda)  # Initial regularization strength

    print("\n--- Starting Targeted Update with Regularized Least Squares ---")
    print("Iter | mean(r) | MSE | mean(u1-u0) | epsilon | lambda")
    print("------------------------------------------------------")

    # For early stopping
    best_mse = initial_mse
    stagnation_counter = 0
    lambda_reg = init_lambda

    for it in range(max_iter):
        # Use projection_step with regularization
        r_np, best_epsilon = projection_step_with_regularization(
            net, X_data, A_data, Y_data, p_data,
            params_targ, lambda_reg=lambda_reg, device=device,
            prune_threshold=prune_threshold,
            verbose = False
        )

       
        # Track metrics after each iteration
        current_mse, current_ate,mu1,mu0= calculate_metrics(net, X_data, A_data, Y_data, device)

        mean_r = np.mean(r_np)
        u_diff = mu1 - mu0
        mean_u_diff = np.mean(u_diff)
        adjusted_data = r_np + u_diff - mean_u_diff
        std_r = np.std(adjusted_data, ddof=1) if len(adjusted_data) > 1 else 0.0
        tau = std_r / (np.sqrt(n_data) * np.log(n_data + 1))


        mse_values.append(current_mse)
        ate_values.append(current_ate)
        mean_r_values.append(mean_r)
        epsilon_values.append(best_epsilon)
        lambda_values.append(lambda_reg)

        # Check for improvement
        if current_mse < best_mse :  # Meaningful improvement
            best_mse = current_mse
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Adaptive regularization strategy
        if stagnation_counter >= 2:
            # If we're stuck, increase regularization to escape local minima
            lambda_reg = min(lambda_reg * 1.5, 1.0)
            if verbose:
                print(f"Increasing lambda to {lambda_reg:.6f} to escape local minimum")
        elif it > 0 and epsilon_values[-1] == 0.0 and epsilon_values[-2] == 0.0:
            # If epsilon is consistently zero, try decreasing regularization
            lambda_reg = max(lambda_reg * 0.5, 1e-5)
            print(f"Decreasing lambda to {lambda_reg:.6f} to allow more movement")

        if verbose:
            print(f"{it:4d} | {mean_r:+.4f} | {current_mse:.4f} | {current_ate:.4f} | {best_epsilon:+.6f} | {lambda_reg:.6f}")

        # Stop if we've stagnated for too long
        if stagnation_counter >= patience:
            if verbose:
                print("\nStopped early: No improvement for", patience, "iterations")
            break

        # Original stopping criterion based on mean residual
        if abs(mean_r) < min(tau, 0.001):
            if verbose:
                print("\nStopped early: |mean(r)| < threshold")
                print(f"Final mean(r) = {mean_r:.6f}, threshold = {tau:.6f}")
            break

    print("----------------------------------------------")
    net.eval()

    # Return metrics including regularization values
    return net, {
        'mse_values': mse_values,
        'ate_values': ate_values,
        'mean_r_values': mean_r_values,
        'epsilon_values': epsilon_values,
        'lambda_values': lambda_values
    }

###############################################################################
# 10) PLOT METRICS FUNCTION WITH EPSILON AND LAMBDA TRACKING
###############################################################################
def plot_metrics_with_regularization(metrics, title="Targeting Metrics with Regularization"):
    """
    Plot the metrics tracked during targeting, including epsilon values and regularization strength
    """
    # Create figure with five subplots
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 18), sharex=True)

    # Plot MSE
    ax1.plot(metrics['mse_values'], 'b-', label='MSE')
    ax1.set_ylabel('MSE')
    ax1.set_title(f'{title} - MSE')
    ax1.grid(True)
    ax1.legend()

    # Plot mean(u1-u0)
    ax2.plot(metrics['ate_values'], 'r-', label='mean(u1-u0)')
    ax2.axhline(y=29.5, color='g', linestyle='--', label='True ATE')
    ax2.set_ylabel('mean(u1-u0)')
    ax2.set_title(f'{title} - Model-based ATE')
    ax2.legend()
    ax2.grid(True)

    # Plot mean(r)
    ax3.plot(metrics['mean_r_values'][1:], 'k-', label='mean(r)')  # Skip first value which is placeholder
    ax3.axhline(y=0, color='g', linestyle='--')
    ax3.set_ylabel('mean(r)')
    ax3.set_title(f'{title} - mean(r)')
    ax3.grid(True)

    # Plot selected epsilon values
    ax4.plot(metrics['epsilon_values'][1:], 'mo-', label='Epsilon')  # Skip first value which is placeholder
    ax4.axhline(y=0, color='g', linestyle='--')
    ax4.set_ylabel('Epsilon')
    ax4.set_title(f'{title} - Selected Epsilon Values')
    ax4.grid(True)

    # Plot regularization strength values
    ax5.plot(metrics['lambda_values'], 'co-', label='Lambda')
    ax5.set_yscale('log')  # Log scale for lambda values
    ax5.set_ylabel('Lambda')
    ax5.set_xlabel('Iteration')
    ax5.set_title(f'{title} - Regularization Strength')
    ax5.grid(True)

    plt.tight_layout()
    return fig

###############################################################################
# 11) DOUBLE ROBUST ATE ESTIMATION
###############################################################################
def dr_ate(Y, A, p, mu0, mu1):
    """
    Double-robust ATE estimator:
      psi_i = (mu1 - mu0) + (A/p)*(Y - mu1) - ((1-A)/(1-p))*(Y - mu0)
    Returns the estimated ATE and standard error.
    """
    term_diff = mu1 - mu0
    term_res = (A / p) * (Y - mu1) - ((1. - A) / (1. - p)) * (Y - mu0)
    psi_vals = term_diff + term_res
    psi_est = np.mean(psi_vals)
    inf_func = psi_vals - psi_est
    varIF = np.var(inf_func, ddof=1)
    se = np.sqrt(varIF / len(Y))
    return psi_est, se, np.mean(inf_func), varIF

def single_step_tmle(Y, A, p, mu0, mu1, true_ate):
    """
    Perform a single-step TMLE update and return results.

    Args:
        Y (array): Outcome
        A (array): Treatment
        p (array): Propensity score
        mu0 (array): Predicted outcome under control
        mu1 (array): Predicted outcome under treatment
        true_ate (float): True ATE (for coverage check)

    Returns:
        dict with model_ate_tmle, model_se_tmle, model_cover_tmle, model_varIF_tmle
    """
    # Compute clever covariate
    clever_covariate = (A / p) - ((1 - A) / (1 - p))
    H = clever_covariate.reshape(-1, 1)  # shape (n,1)

    # Solve for epsilon via least squares
    residual = A * (Y - mu1) + (1-A) * (Y - mu0)
    epsilon_tmle = np.linalg.lstsq(H, residual, rcond=None)[0][0]

    # Update mu0 and mu1
    mu0_tmle = mu0 - epsilon_tmle * (1 / (1 - p))
    mu1_tmle = mu1 + epsilon_tmle * (1 / p)

    # New plug-in ATE
    model_ate_tmle = np.mean(mu1_tmle - mu0_tmle)

    # Influence function
    term_diff_tmle = mu1_tmle - mu0_tmle
    term_res_tmle = (A / p) * (Y - mu1_tmle) - ((1. - A) / (1. - p)) * (Y - mu0_tmle)
    psi_vals_tmle = term_diff_tmle + term_res_tmle
    psi_est_tmle = np.mean(term_diff_tmle)
    inf_func_tmle = psi_vals_tmle - psi_est_tmle
    varIF_tmle = np.var(inf_func_tmle, ddof=1)
    se_tmle = np.sqrt(varIF_tmle / len(Y))

    # Confidence interval
    lower_tmle = model_ate_tmle - 1.96 * se_tmle
    upper_tmle = model_ate_tmle + 1.96 * se_tmle
    cover_tmle = (true_ate >= lower_tmle) and (true_ate <= upper_tmle)

    return model_ate_tmle, se_tmle, cover_tmle, np.mean(inf_func_tmle), varIF_tmle

def coverage_and_ci(est, se, true_val):
    """
    Calculate confidence interval and check if it covers the true value
    """
    lower = est - 1.96 * se
    upper = est + 1.96 * se
    covered = (true_val >= lower) and (true_val <= upper)
    return covered, lower, upper

def model_ate_ci(Y, A, p, mu0, mu1, true_ate):
    """
    Double-robust ATE estimator:
      psi_i = (mu1 - mu0) + (A/p)*(Y - mu1) - ((1-A)/(1-p))*(Y - mu0)
    Returns the estimated ATE and standard error.
    """
    term_diff = mu1 - mu0
    term_res = (A / p) * (Y - mu1) - ((1. - A) / (1. - p)) * (Y - mu0)
    psi_vals = term_diff + term_res
    psi_est = np.mean(term_diff)
    inf_func = psi_vals - psi_est
    varIF = np.var(inf_func, ddof=1)
    se = np.sqrt(varIF / len(Y))
    lower = psi_est - 1.96 * se
    upper = psi_est + 1.96 * se
    covered = (true_ate >= lower) and (true_ate <= upper)
    return psi_est, se, covered, lower, upper, np.mean(inf_func), varIF

###############################################################################
# 3.1) NEW: TRAINER WITH TARGETING REGULARIZATION
###############################################################################
def train_with_targeting_and_early_stopping(net, X_train, A_train, Y_train, 
                                           X_val, A_val, Y_val,
                                           max_epochs=200, batch_size=128, 
                                           lr=1e-3, weight_decay=1e-5,
                                           targeting_frequency=10, 
                                           targeting_lambda=0.01,
                                           targeting_params=None,
                                           patience=10,
                                           device='cpu',
                                           verbose=False):
    """
    Train with targeting regularization and early stopping.
    
    Returns:
        trained_net: The best model based on validation loss
        metrics: Dictionary of training metrics
    """
    net = net.to(device)
    
    # Prepare data
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    A_train_t = torch.tensor(A_train, dtype=torch.float32).view(-1, 1).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1).to(device)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    A_val_t = torch.tensor(A_val, dtype=torch.float32).view(-1, 1).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).view(-1, 1).to(device)
    
    # Create data loader
    train_ds = torch.utils.data.TensorDataset(X_train_t, A_train_t, Y_train_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Select parameters for standard training (excluding ATE parameter)
    params_std = [p for n, p in net.named_parameters() if n != 'ate']
    opt = optim.Adam(params_std, lr=lr, weight_decay=weight_decay)
    
    # Select parameters for targeting if not specified
    if targeting_params is None:
        targeting_params = []
        for nm, param in net.named_parameters():
            if nm in ['mu0_out.weight', 'mu1_out.weight']:
                targeting_params.append(param)
    
    # For early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Lists to track metrics
    train_losses = []
    val_losses = []
    epochs = []
    targeting_updates = []
    targeting_epsilons = []
    targeting_mean_r = []
    model_ates = []
    
    for epoch in range(max_epochs):
        # ===== TRAINING PHASE =====
        net.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for xb, ab, yb in train_loader:
            p_hat, mu0, mu1, _ = net(xb)
            bce_loss = F.binary_cross_entropy(p_hat, ab)
            mask0 = (ab < 0.5)
            mask1 = (ab > 0.5)
            mse0 = torch.mean((yb[mask0] - mu0[mask0])**2) if mask0.any() else 0.0
            mse1 = torch.mean((yb[mask1] - mu1[mask1])**2) if mask1.any() else 0.0
            loss = bce_loss + mse0 + mse1
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_std, 5.0)
            opt.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        train_losses.append(avg_train_loss)
        
        # ===== VALIDATION PHASE =====
        net.eval()
        with torch.no_grad():
            p_val, mu0_val, mu1_val, _ = net(X_val_t)
            bce_val = F.binary_cross_entropy(p_val, A_val_t)
            mask0_val = (A_val_t < 0.5)
            mask1_val = (A_val_t > 0.5)
            mse0_val = torch.mean((Y_val_t[mask0_val] - mu0_val[mask0_val])**2) if mask0_val.any() else 0.0
            mse1_val = torch.mean((Y_val_t[mask1_val] - mu1_val[mask1_val])**2) if mask1_val.any() else 0.0
            val_loss = bce_val + mse0_val + mse1_val
        
        val_losses.append(val_loss.item())
        epochs.append(epoch)
        
        # Get current model-based ATE for tracking
        with torch.no_grad():
            _, mu0_train, mu1_train, _ = net(X_train_t)
            model_ate = torch.mean(mu1_train - mu0_train).item()
        
        model_ates.append(model_ate)
        
        # ===== EARLY STOPPING CHECK =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(net.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # ===== TARGETING UPDATE =====
        # Only apply targeting update after initial training has started and on periodic intervals
        targeting_applied = False
        mean_r = 0.0
        epsilon = 0.0
        
        if epoch > 0 and epoch % targeting_frequency == 0:
            targeting_applied = True
            if verbose:
                print(f"[Epoch {epoch}] Applying targeting update...")
            
            # Get current propensity score predictions
            with torch.no_grad():
                p_hat_train, _, _, _ = net(X_train_t)
                p_train = p_hat_train.cpu().numpy().ravel()
            
            # Apply the targeting update on training data only
            r_np, best_epsilon = projection_step_with_regularization(
                net, X_train, A_train, Y_train, p_train,
                targeting_params, lambda_reg=targeting_lambda, 
                device=device, prune_threshold=0, verbose=False
            )
            
            mean_r = np.mean(r_np)
            epsilon = best_epsilon
            
            if verbose:
                print(f"[Epoch {epoch}] Targeting update with epsilon={best_epsilon:.6f}, mean(r)={mean_r:.6f}")
        
        targeting_updates.append(targeting_applied)
        targeting_epsilons.append(epsilon)
        targeting_mean_r.append(mean_r)
        
        # Report progress
        if verbose and (epoch % 10 == 0 or epoch == max_epochs - 1 or targeting_applied):
            print(f"[Epoch {epoch}] Train loss: {avg_train_loss:.4f}, Val loss: {val_loss.item():.4f}, ATE: {model_ate:.4f}")
    
    # Load the best model based on validation performance
    net.load_state_dict(best_model_state)
    
    # Collect metrics
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs': epochs,
        'targeting_updates': targeting_updates,
        'targeting_epsilons': targeting_epsilons,
        'targeting_mean_r': targeting_mean_r,
        'model_ates': model_ates,
        'best_epoch': epochs[val_losses.index(min(val_losses))],
        'best_val_loss': min(val_losses)
    }
    
    return net, metrics

###############################################################################
# NEW: COMBINED APPROACH - TRAINING WITH TARGETING + POST-TRAINING TARGETING
###############################################################################
def combined_training_and_post_targeting(net, X_train, A_train, Y_train, 
                                         X_val, A_val, Y_val,
                                         X_target, A_target, Y_target,
                                         max_epochs=200, batch_size=128, 
                                         lr=1e-3, weight_decay=1e-5,
                                         targeting_frequency=10, 
                                         targeting_lambda=0.01,
                                         post_targeting_max_iter=50,
                                         post_targeting_lambda=0.01,
                                         layers_full=False,
                                         patience=10,
                                         device='cpu',
                                         verbose=False):
    """
    Combined approach: First use training with targeting regularization,
    then apply post-training targeting.
    
    Returns:
        trained_net: The final model after both stages
        metrics: Dictionary of training metrics from both stages
    """
    # Select parameters based on layers_full flag
    if layers_full:
        targeting_params = []
        for nm, param in net.named_parameters():
            if nm in ['mu0_l1.weight', 'mu0_out.weight', 'mu1_l1.weight', 'mu1_out.weight']:
                targeting_params.append(param)
    else:
        targeting_params = []
        for nm, param in net.named_parameters():
            if nm in ['mu0_out.weight', 'mu1_out.weight']:
                targeting_params.append(param)
    
    # Stage 1: Training with targeting regularization
    print(f"\n--- STAGE 1: Training with {'All-Layers' if layers_full else 'Last-Layer'} Targeting Regularization ---")
    net, metrics_train = train_with_targeting_and_early_stopping(
        net, X_train, A_train, Y_train, X_val, A_val, Y_val,
        max_epochs=max_epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        targeting_frequency=targeting_frequency, targeting_lambda=targeting_lambda,
        targeting_params=targeting_params, patience=patience,
        device=device, verbose=verbose
    )
    
    print(f"Stage 1 completed at epoch {metrics_train['best_epoch']}, val loss: {metrics_train['best_val_loss']:.4f}")
    
    # Predict propensity scores for targeting set
    X_target_t = torch.tensor(X_target, dtype=torch.float32).to(device)
    with torch.no_grad():
        p_target, _, _, _ = net(X_target_t)
        p_target = p_target.cpu().numpy().ravel()
    
    # Stage 2: Post-training targeting
    print(f"\n--- STAGE 2: Post-Training {'All-Layers' if layers_full else 'Last-Layer'} Targeting ---")
    net, metrics_post = targeted_update_with_regularization(
        net, X_target, A_target, Y_target, p_target,
        max_iter=post_targeting_max_iter, device=device,
        prune_threshold=0, verbose=verbose,
        patience=patience, init_lambda=post_targeting_lambda,
        layers_full=layers_full
    )
    
    # Combine metrics from both stages
    combined_metrics = {
        'train_metrics': metrics_train,
        'post_metrics': metrics_post,
        'approach': 'combined',
        'layers_full': layers_full
    }
    
    return net, combined_metrics

###############################################################################
# NEW: TRAINER WITH TMLE-STYLE LOSS FOR TRAINING
###############################################################################
def train_with_tmle_style_loss(net, X_train, A_train, Y_train, 
                              X_val, A_val, Y_val,
                              max_epochs=200, batch_size=128, 
                              lr=1e-3, weight_decay=1e-5,
                              tmle_weight=0.1,
                              patience=10,
                              device='cpu',
                              verbose=False):
    """
    Train with TMLE-style loss added to the standard loss function.
    The TMLE-style loss is: (Y - Q - epsilon * H)^2 where H is the clever covariate.
    
    Args:
        net: DragonNet model
        X_train, A_train, Y_train: Training data
        X_val, A_val, Y_val: Validation data
        max_epochs: Maximum number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: Weight decay for regularization
        tmle_weight: Weight for the TMLE-style loss component
        patience: Early stopping patience
        device: Computing device
        verbose: Whether to print progress
        
    Returns:
        trained_net: The best model based on validation loss
        metrics: Dictionary of training metrics
    """
    net = net.to(device)
    
    # Prepare data
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    A_train_t = torch.tensor(A_train, dtype=torch.float32).view(-1, 1).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1).to(device)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    A_val_t = torch.tensor(A_val, dtype=torch.float32).view(-1, 1).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).view(-1, 1).to(device)
    
    # Create data loader
    train_ds = torch.utils.data.TensorDataset(X_train_t, A_train_t, Y_train_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Select parameters for training (excluding ATE parameter)
    params_std = [p for n, p in net.named_parameters() if n != 'ate']
    opt = optim.Adam(params_std, lr=lr, weight_decay=weight_decay)
    
    # For early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Lists to track metrics
    train_losses = []
    train_std_losses = []
    train_tmle_losses = []
    val_losses = []
    val_std_losses = []
    val_tmle_losses = []
    epsilon_values = []
    model_ates = []
    epochs = []
    
    for epoch in range(max_epochs):
        # ===== TRAINING PHASE =====
        net.train()
        epoch_loss = 0.0
        epoch_std_loss = 0.0
        epoch_tmle_loss = 0.0
        n_batches = 0
        
        for xb, ab, yb in train_loader:
            # Forward pass
            p_hat, mu0, mu1, _ = net(xb)
            
            # Standard loss
            bce_loss = F.binary_cross_entropy(p_hat, ab)
            mask0 = (ab < 0.5)
            mask1 = (ab > 0.5)
            mse0 = torch.mean((yb[mask0] - mu0[mask0])**2) if mask0.any() else 0.0
            mse1 = torch.mean((yb[mask1] - mu1[mask1])**2) if mask1.any() else 0.0
            std_loss = bce_loss + mse0 + mse1
            
            # TMLE-style loss computation
            # Compute the clever covariate H
            H = ab / (p_hat + 1e-7) - (1 - ab) / (1 - p_hat + 1e-7)
            
            # Compute predicted outcome Q based on treatment status
            Q = ab * mu1 + (1 - ab) * mu0
            
            # Compute the residual Y - Q
            residual = yb - Q
            
            # Estimate epsilon (fit using current batch)
            # Using a closed-form solution for the linear regression
            with torch.no_grad():
                # Solve the normal equation: (H^T * H) * epsilon = H^T * residual
                HTH = torch.sum(H * H)
                HTr = torch.sum(H * residual)
                # Prevent division by zero
                if HTH > 1e-10:
                    epsilon = HTr / HTH
                else:
                    epsilon = torch.tensor(0.0).to(device)
            
            # Compute TMLE-style loss: (Y - Q - epsilon * H)^2
            tmle_loss = torch.mean((residual - epsilon * H) ** 2)
            
            # Combined loss
            loss = std_loss + tmle_weight * tmle_loss
            
            # Backward pass and optimization
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_std, 5.0)
            opt.step()
            
            epoch_loss += loss.item()
            epoch_std_loss += std_loss.item()
            epoch_tmle_loss += tmle_loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        avg_train_std_loss = epoch_std_loss / n_batches if n_batches > 0 else 0.0
        avg_train_tmle_loss = epoch_tmle_loss / n_batches if n_batches > 0 else 0.0
        
        train_losses.append(avg_train_loss)
        train_std_losses.append(avg_train_std_loss)
        train_tmle_losses.append(avg_train_tmle_loss)
        
        # ===== VALIDATION PHASE =====
        net.eval()
        with torch.no_grad():
            # Forward pass
            p_val, mu0_val, mu1_val, _ = net(X_val_t)
            
            # Standard loss
            bce_val = F.binary_cross_entropy(p_val, A_val_t)
            mask0_val = (A_val_t < 0.5)
            mask1_val = (A_val_t > 0.5)
            mse0_val = torch.mean((Y_val_t[mask0_val] - mu0_val[mask0_val])**2) if mask0_val.any() else 0.0
            mse1_val = torch.mean((Y_val_t[mask1_val] - mu1_val[mask1_val])**2) if mask1_val.any() else 0.0
            std_loss_val = bce_val + mse0_val + mse1_val
            
            # TMLE-style loss computation for validation
            # Compute the clever covariate H
            H_val = A_val_t / (p_val + 1e-7) - (1 - A_val_t) / (1 - p_val + 1e-7)
            
            # Compute predicted outcome Q based on treatment status
            Q_val = A_val_t * mu1_val + (1 - A_val_t) * mu0_val
            
            # Compute the residual Y - Q
            residual_val = Y_val_t - Q_val
            
            # Estimate epsilon (fit using validation set)
            # Using a closed-form solution for the linear regression
            HTH_val = torch.sum(H_val * H_val)
            HTr_val = torch.sum(H_val * residual_val)
            # Prevent division by zero
            if HTH_val > 1e-10:
                epsilon_val = HTr_val / HTH_val
            else:
                epsilon_val = torch.tensor(0.0).to(device)
            
            # Compute TMLE-style loss: (Y - Q - epsilon * H)^2
            tmle_loss_val = torch.mean((residual_val - epsilon_val * H_val) ** 2)
            
            # Combined validation loss
            val_loss = std_loss_val + tmle_weight * tmle_loss_val
            
            # Track current epsilon and ATE
            model_ate = torch.mean(mu1_val - mu0_val).item()
            
        val_losses.append(val_loss.item())
        val_std_losses.append(std_loss_val.item())
        val_tmle_losses.append(tmle_loss_val.item())
        epsilon_values.append(epsilon_val.item())
        model_ates.append(model_ate)
        epochs.append(epoch)
        
        # ===== EARLY STOPPING CHECK =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(net.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Report progress
        if verbose and epoch % 10 == 0:
            print(f"[Epoch {epoch}] Train loss: {avg_train_loss:.4f} (std: {avg_train_std_loss:.4f}, tmle: {avg_train_tmle_loss:.4f})")
            print(f"  Val loss: {val_loss.item():.4f}, epsilon: {epsilon_val.item():.4f}, ATE: {model_ate:.4f}")
    
    # Load the best model based on validation performance
    net.load_state_dict(best_model_state)
    
    # Collect metrics
    metrics = {
        'train_losses': train_losses,
        'train_std_losses': train_std_losses,
        'train_tmle_losses': train_tmle_losses,
        'val_losses': val_losses,
        'val_std_losses': val_std_losses,
        'val_tmle_losses': val_tmle_losses,
        'epsilon_values': epsilon_values,
        'model_ates': model_ates,
        'epochs': epochs,
        'best_epoch': epochs[val_losses.index(min(val_losses))],
        'best_val_loss': min(val_losses),
        'final_epsilon': epsilon_values[-1]
    }
    
    return net, metrics

###############################################################################
# NEW: TRAINER WITH TMLE-STYLE LOSS + FINAL TMLE UPDATE
###############################################################################
def train_with_tmle_style_loss_and_update(net, X_train, A_train, Y_train, 
                                         X_val, A_val, Y_val,
                                         max_epochs=200, batch_size=128, 
                                         lr=1e-3, weight_decay=1e-5,
                                         tmle_weight=0.1,
                                         patience=10,
                                         device='cpu',
                                         verbose=False):
    """
    Train with TMLE-style loss added to the standard loss function,
    then perform a final TMLE update step to get Q* = Q + epsilon*H.
    
    Returns:
        trained_net: The updated model with TMLE correction
        metrics: Dictionary of training metrics
        epsilon_final: The final epsilon value from the TMLE update
    """
    # First, train with TMLE-style loss
    net, metrics = train_with_tmle_style_loss(
        net, X_train, A_train, Y_train, X_val, A_val, Y_val,
        max_epochs, batch_size, lr, weight_decay, tmle_weight,
        patience, device, verbose
    )
    
    # Now perform the final TMLE update on the combined train+val data
    X_combined = np.vstack([X_train, X_val])
    A_combined = np.concatenate([A_train, A_val])
    Y_combined = np.concatenate([Y_train, Y_val])
    
    X_combined_t = torch.tensor(X_combined, dtype=torch.float32).to(device)
    A_combined_t = torch.tensor(A_combined, dtype=torch.float32).view(-1, 1).to(device)
    Y_combined_t = torch.tensor(Y_combined, dtype=torch.float32).view(-1, 1).to(device)
    
    net.eval()
    with torch.no_grad():
        # Forward pass
        p_hat, mu0, mu1, _ = net(X_combined_t)
        
        # Compute the clever covariate H
        H = A_combined_t / (p_hat + 1e-7) - (1 - A_combined_t) / (1 - p_hat + 1e-7)
        
        # Compute original predicted outcome Q based on treatment status
        Q = A_combined_t * mu1 + (1 - A_combined_t) * mu0
        
        # Compute the residual Y - Q
        residual = Y_combined_t - Q
        
        # Estimate epsilon using closed-form solution
        HTH = torch.sum(H * H)
        HTr = torch.sum(H * residual)
        
        if HTH > 1e-10:
            epsilon_final = (HTr / HTH).item()
        else:
            epsilon_final = 0.0
            
        if verbose:
            print(f"Final TMLE update epsilon: {epsilon_final:.6f}")
            
        # Compute updated outcomes Q* = Q + epsilon*H
        mu0_updated = mu0 - epsilon_final * (1.0 / (1.0 - p_hat + 1e-7))
        mu1_updated = mu1 + epsilon_final * (1.0 / (p_hat + 1e-7))
        
        # Compute updated ATE
        orig_ate = torch.mean(mu1 - mu0).item()
        updated_ate = torch.mean(mu1_updated - mu0_updated).item()
        
        if verbose:
            print(f"Original ATE: {orig_ate:.4f}, Updated ATE: {updated_ate:.4f}")
    
    # Create a modified network with the updated outcomes
    # This is a bit of a hack - we're creating a special network class that will
    # apply the TMLE correction during the forward pass
    
    class TMLECorrectedDragonNet(nn.Module):
        def __init__(self, base_net, epsilon):
            super().__init__()
            self.base_net = base_net
            self.epsilon = epsilon
            
        def forward(self, x):
            p_hat, mu0, mu1, ate = self.base_net(x)
            
            # Apply TMLE correction
            mu0_updated = mu0 - self.epsilon * (1.0 / (1.0 - p_hat + 1e-7))
            mu1_updated = mu1 + self.epsilon * (1.0 / (p_hat + 1e-7))
            
            # Update ATE parameter
            ate_updated = torch.mean(mu1_updated - mu0_updated)
            
            return p_hat, mu0_updated, mu1_updated, ate_updated
    
    tmle_corrected_net = TMLECorrectedDragonNet(net, epsilon_final).to(device)
    
    # Add TMLE update info to metrics
    metrics['epsilon_final'] = epsilon_final
    metrics['orig_ate'] = orig_ate
    metrics['updated_ate'] = updated_ate
    
    return tmle_corrected_net, metrics, epsilon_final

###############################################################################
# NEW: PLOT TMLE-STYLE LOSS METRICS
###############################################################################
def plot_tmle_style_loss_metrics(metrics, title="TMLE-style Loss Training Metrics"):
    """
    Plot the metrics from TMLE-style loss training
    """
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Epochs
    epochs = metrics['epochs']
    
    # Plot losses
    ax1.plot(epochs, metrics['train_losses'], 'b-', label='Train Total Loss')
    ax1.plot(epochs, metrics['val_losses'], 'r-', label='Val Total Loss')
    ax1.set_ylabel('Total Loss')
    ax1.set_title(f'{title} - Total Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot component losses
    ax2.plot(epochs, metrics['train_std_losses'], 'b-', label='Train Std Loss')
    ax2.plot(epochs, metrics['val_std_losses'], 'r-', label='Val Std Loss')
    ax2.plot(epochs, metrics['train_tmle_losses'], 'b--', label='Train TMLE Loss')
    ax2.plot(epochs, metrics['val_tmle_losses'], 'r--', label='Val TMLE Loss')
    ax2.set_ylabel('Component Losses')
    ax2.set_title(f'{title} - Component Losses')
    ax2.grid(True)
    ax2.legend()
    
    # Plot model ATE
    ax3.plot(epochs, metrics['model_ates'], 'r-', label='Model ATE')
    ax3.axhline(y=29.5, color='g', linestyle='--', label='True ATE')
    ax3.set_ylabel('ATE')
    ax3.set_xlabel('Epoch')
    ax3.set_title(f'{title} - Model ATE')
    ax3.grid(True)
    ax3.legend()
    
    # Plot epsilon values
    ax4.plot(epochs, metrics['epsilon_values'], 'mo-', label='Epsilon Values')
    ax4.axhline(y=0, color='g', linestyle='--')
    ax4.set_ylabel('Epsilon')
    ax4.set_xlabel('Epoch')
    ax4.set_title(f'{title} - Epsilon Values')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    return fig

###############################################################################
# SUMMARY FUNCTION FOR TMLE APPROACHES RESULTS
###############################################################################
def summarize_tmle_approaches_results(path):
    """
    Create a summary of results from TMLE approaches experiments
    """
    try:
        df = pd.read_csv(path)
    except:
        print(f"Error: Could not load results from {path}")
        return None
    
    summary_rows = []
    
    # Define all estimator prefixes we care about
    estimator_prefixes = [
        ('model', 'std'),
        ('model', 'post_last'),
        ('model', 'post_full'),
        ('model', 'ttr_last'),
        ('model', 'ttr_full'),
        ('model', 'combined_last'),
        ('model', 'combined_full'),
        ('model', 'tmle_loss'),
        ('model', 'tmle_update'),
        ('model', 'tmle'),
        ('dr', 'std')
    ]
    
    for est, stage in estimator_prefixes:
        prefix = f"{est}_ate_{stage}"
        cover_col = f"{est}_cover_{stage}"
        se_col = f"{est}_se_{stage}"
        meanIF_col = f"{est}_meanIF_{stage}"
        varIF_col = f"{est}_varIF_{stage}"
        
        if prefix not in df.columns:
            print(f"Warning: Column {prefix} not found in results")
            continue  # Skip if missing
        
        try:
            est_vals = df[prefix]
            true_vals = df['true_ate']
            se = df[se_col]
            covers = df[cover_col]
            meanIF = df[meanIF_col]
            varIF = df[varIF_col]
        except KeyError as e:
            print(f"Warning: Missing column: {e}")
            continue  # skip if any column is missing
        
        # Compute metrics
        bias = np.mean(est_vals - true_vals)
        variance = np.var(est_vals - true_vals, ddof=1)
        mse = bias**2 + variance
        rmse = np.sqrt(mse)
        coverage = np.mean(covers)
        mean_if_avg = np.mean(meanIF)
        var_if_avg = np.mean(varIF)
        se_avg = np.mean(se)
        
        summary_rows.append({
            'estimator': f'{est}_{stage}',
            'bias': bias,
            'variance': variance,
            'mse': mse,
            'rmse': rmse,
            'coverage': coverage,
            'se_ci_avg': se_avg,
            'mean_IF': mean_if_avg,
            'var_IF': var_if_avg
        })
    
    if not summary_rows:
        print("Warning: No valid data found for summary")
        return None
        
    summary_df = pd.DataFrame(summary_rows)
    
    # Add a description column for clarity
    descriptions = {
        'model_std': 'Standard Training',
        'model_post_last': 'Post-Training Targeting (Last Layer)',
        'model_post_full': 'Post-Training Targeting (All Layers)',
        'model_ttr_last': 'Training with Targeting (Last Layer)',
        'model_ttr_full': 'Training with Targeting (All Layers)',
        'model_combined_last': 'Combined Approach (Last Layer)',
        'model_combined_full': 'Combined Approach (All Layers)',
        'model_tmle_loss': 'Training with TMLE-style Loss',
        'model_tmle_update': 'Training with TMLE-style Loss + TMLE Update',
        'model_tmle': 'Standard + TMLE Update',
        'dr_std': 'Double Robust ATE'
    }
    
    summary_df['description'] = summary_df['estimator'].map(descriptions)
    
    # Reorder columns to put description first
    cols = summary_df.columns.tolist()
    cols.remove('description')
    summary_df = summary_df[['description'] + cols]
    
    return summary_df

###############################################################################
# RUN EXPERIMENT INCLUDING TMLE-STYLE LOSS APPROACHES
###############################################################################
def run_experiment_with_tmle_approaches(X, A, Y, p_true, true_ate, device='cpu',
                                       init_lambda=0.01, patience=5, 
                                       train_ratio=0.6, val_ratio=0.2,
                                       targeting_frequency=10,
                                       targeting_lambda=0.01,
                                       post_targeting_max_iter=50,
                                       post_targeting_lambda=0.01,
                                       tmle_weight=0.1,
                                       verbose=False):
    """
    Run experiment with three-fold data splitting, including TMLE-style loss approaches:
    - Estimators 1-7: Original estimators from previous implementation
    - Estimator 8: Training with TMLE-style loss
    - Estimator 9: Training with TMLE-style loss + TMLE update
    """
    # Create three-fold split
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    # Calculate split sizes based on ratios
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    target_idx = indices[train_size+val_size:]
    
    # Create the three datasets
    X_train, A_train, Y_train = X[train_idx], A[train_idx], Y[train_idx]
    X_val, A_val, Y_val = X[val_idx], A[val_idx], Y[val_idx]
    #X_target, A_target, Y_target = X[target_idx], A[target_idx], Y[target_idx]
    #p_true_target = p_true[target_idx] if p_true is not None else None
    
    X_target, A_target, Y_target = X, A, Y

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Targeting set size: {len(X_target)}")
    
    # ESTIMATOR 1: Standard Training with Early Stopping
    print("\n--- ESTIMATOR 1: Standard Training with Early Stopping ---")
    net_std = DragonNet(xdim=X.shape[1], hidden_shared=64, hidden_outcome=64)
    trainer_std = DragonTrainer(net_std, device=device)
    
    # Custom training loop with validation-based early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(200):  # Max epochs
        # Train for one epoch
        train_loss = trainer_std.train_epoch(X_train, A_train, Y_train, batch_size=128)
        
        # Evaluate on validation set
        val_loss = trainer_std.evaluate(X_val, A_val, Y_val)
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(net_std.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0 and verbose:
            print(f"Epoch {epoch}, train loss: {train_loss:.4f}, validation loss: {val_loss:.4f}")
    
    # Load the best model based on validation performance
    net_std.load_state_dict(best_model_state)
    print(f"Loaded best standard model with validation loss: {best_val_loss:.4f}")
    
    # Get predictions for the targeting set
    p_init, mu0_init, mu1_init = trainer_std.predict(X_target)
    
    # ESTIMATOR 2: Post-training targeting (last layer)
    print("\n--- ESTIMATOR 2: Standard Post-Training Targeting (Last Layer) ---")
    net_post_last = deepcopy(net_std)
    net_post_last, metrics_post_last = targeted_update_with_regularization(
        net_post_last, X_target, A_target, Y_target, p_init,
        max_iter=post_targeting_max_iter, device=device,
        prune_threshold=0,
        verbose=True,
        patience=patience,
        init_lambda=post_targeting_lambda,
        layers_full=False
    )
    
    # ESTIMATOR 3: Post-training targeting (all layers)
    print("\n--- ESTIMATOR 3: Standard Post-Training Targeting (All Layers) ---")
    net_post_full = deepcopy(net_std)
    net_post_full, metrics_post_full = targeted_update_with_regularization(
        net_post_full, X_target, A_target, Y_target, p_init,
        max_iter=post_targeting_max_iter, device=device,
        prune_threshold=0,
        verbose=True,
        patience=patience,
        init_lambda=post_targeting_lambda,
        layers_full=True
    )
    
    # ESTIMATOR 4: Training with last-layer targeting regularization
    print("\n--- ESTIMATOR 4: Training with Last-Layer Targeting Regularization ---")
    net_ttr_last = DragonNet(xdim=X.shape[1], hidden_shared=64, hidden_outcome=64)
    
    # Select last-layer parameters
    last_layer_params = []
    for nm, param in net_ttr_last.named_parameters():
        if nm in ['mu0_out.weight', 'mu1_out.weight']:
            last_layer_params.append(param)
    
    # Train with targeting and early stopping
    net_ttr_last, metrics_ttr_last = train_with_targeting_and_early_stopping(
        net_ttr_last, X_train, A_train, Y_train, X_val, A_val, Y_val,
        max_epochs=200, batch_size=128, lr=1e-3, weight_decay=1e-5,
        targeting_frequency=targeting_frequency, targeting_lambda=targeting_lambda,
        targeting_params=last_layer_params, patience=patience,
        device=device, verbose=verbose
    )
    
    print(f"Training with last-layer targeting completed at epoch {metrics_ttr_last['best_epoch']}, val loss: {metrics_ttr_last['best_val_loss']:.4f}")
    
    # ESTIMATOR 5: Training with all-layers targeting regularization
    print("\n--- ESTIMATOR 5: Training with All-Layers Targeting Regularization ---")
    net_ttr_full = DragonNet(xdim=X.shape[1], hidden_shared=64, hidden_outcome=64)
    
    # Select all outcome layers parameters
    full_layer_params = []
    for nm, param in net_ttr_full.named_parameters():
        if nm in ['mu0_l1.weight', 'mu0_out.weight', 'mu1_l1.weight', 'mu1_out.weight']:
            full_layer_params.append(param)
    
    # Train with targeting and early stopping
    net_ttr_full, metrics_ttr_full = train_with_targeting_and_early_stopping(
        net_ttr_full, X_train, A_train, Y_train, X_val, A_val, Y_val,
        max_epochs=200, batch_size=128, lr=1e-3, weight_decay=1e-5,
        targeting_frequency=targeting_frequency, targeting_lambda=targeting_lambda,
        targeting_params=full_layer_params, patience=patience,
        device=device, verbose=verbose
    )
    
    print(f"Training with all-layers targeting completed at epoch {metrics_ttr_full['best_epoch']}, val loss: {metrics_ttr_full['best_val_loss']:.4f}")
    
    # ESTIMATOR 6: Combined - Training with Last-Layer Targeting + Post-Training Last-Layer Targeting
    print("\n--- ESTIMATOR 6: Combined - Training with Last-Layer Targeting + Post-Training Last-Layer Targeting ---")
    net_combined_last = DragonNet(xdim=X.shape[1], hidden_shared=64, hidden_outcome=64)
    net_combined_last, metrics_combined_last = combined_training_and_post_targeting(
        net_combined_last, X_train, A_train, Y_train, X_val, A_val, Y_val, X_target, A_target, Y_target,
        max_epochs=200, batch_size=128, lr=1e-3, weight_decay=1e-5,
        targeting_frequency=targeting_frequency, targeting_lambda=targeting_lambda,
        post_targeting_max_iter=post_targeting_max_iter, post_targeting_lambda=post_targeting_lambda,
        layers_full=False, patience=patience, device=device, verbose=verbose
    )
    
    # ESTIMATOR 7: Combined - Training with All-Layers Targeting + Post-Training All-Layers Targeting
    print("\n--- ESTIMATOR 7: Combined - Training with All-Layers Targeting + Post-Training All-Layers Targeting ---")
    net_combined_full = DragonNet(xdim=X.shape[1], hidden_shared=64, hidden_outcome=64)
    net_combined_full, metrics_combined_full = combined_training_and_post_targeting(
        net_combined_full, X_train, A_train, Y_train, X_val, A_val, Y_val, X_target, A_target, Y_target,
        max_epochs=200, batch_size=128, lr=1e-3, weight_decay=1e-5,
        targeting_frequency=targeting_frequency, targeting_lambda=targeting_lambda,
        post_targeting_max_iter=post_targeting_max_iter, post_targeting_lambda=post_targeting_lambda,
        layers_full=True, patience=patience, device=device, verbose=verbose
    )
    
    # ESTIMATOR 8: Training with TMLE-style loss
    print("\n--- ESTIMATOR 8: Training with TMLE-style Loss ---")
    net_tmle_loss = DragonNet(xdim=X.shape[1], hidden_shared=64, hidden_outcome=64)
    net_tmle_loss, metrics_tmle_loss = train_with_tmle_style_loss(
        net_tmle_loss, X_train, A_train, Y_train, X_val, A_val, Y_val,
        max_epochs=200, batch_size=128, lr=1e-3, weight_decay=1e-5,
        tmle_weight=tmle_weight, patience=patience,
        device=device, verbose=verbose
    )
    
    print(f"Training with TMLE-style loss completed at epoch {metrics_tmle_loss['best_epoch']}, val loss: {metrics_tmle_loss['best_val_loss']:.4f}")
    
    # ESTIMATOR 9: Training with TMLE-style loss + TMLE update
    print("\n--- ESTIMATOR 9: Training with TMLE-style Loss + TMLE Update ---")
    net_tmle_update = DragonNet(xdim=X.shape[1], hidden_shared=64, hidden_outcome=64)
    net_tmle_update, metrics_tmle_update, epsilon_final = train_with_tmle_style_loss_and_update(
        net_tmle_update, X_train, A_train, Y_train, X_val, A_val, Y_val,
        max_epochs=200, batch_size=128, lr=1e-3, weight_decay=1e-5,
        tmle_weight=tmle_weight, patience=patience,
        device=device, verbose=verbose
    )
    
    print(f"Training with TMLE-style loss + update completed at epoch {metrics_tmle_update['best_epoch']}, final epsilon: {epsilon_final:.6f}")
    
    # Get predictions from all models on the targeting/evaluation set
    X_tensor = torch.tensor(X_target, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # Initial model (post standard training with early stopping)
        p_std, mu0_std, mu1_std, _ = net_std(X_tensor)
        
        # Post-training targeting models
        p_post_last, mu0_post_last, mu1_post_last, _ = net_post_last(X_tensor)
        p_post_full, mu0_post_full, mu1_post_full, _ = net_post_full(X_tensor)
        
        # Training targeting regularization models
        p_ttr_last, mu0_ttr_last, mu1_ttr_last, _ = net_ttr_last(X_tensor)
        p_ttr_full, mu0_ttr_full, mu1_ttr_full, _ = net_ttr_full(X_tensor)
        
        # Combined approach models
        p_combined_last, mu0_combined_last, mu1_combined_last, _ = net_combined_last(X_tensor)
        p_combined_full, mu0_combined_full, mu1_combined_full, _ = net_combined_full(X_tensor)
        
        # TMLE-style loss models
        p_tmle_loss, mu0_tmle_loss, mu1_tmle_loss, _ = net_tmle_loss(X_tensor)
        p_tmle_update, mu0_tmle_update, mu1_tmle_update, _ = net_tmle_update(X_tensor)
    
    # Apply truncation to all propensity scores
    # If first DGP, make truncation less harsh
    if X.shape[1] == 1:
        print("Truncating to 0.005 and 0.995")
        p_std = torch.clamp(p_std, 0.005, 0.995)
        p_post_last = torch.clamp(p_post_last, 0.005, 0.995)
        p_post_full = torch.clamp(p_post_full, 0.005, 0.995)
        p_ttr_last = torch.clamp(p_ttr_last, 0.005, 0.995)
        p_ttr_full = torch.clamp(p_ttr_full, 0.005, 0.995)
        p_combined_last = torch.clamp(p_combined_last, 0.005, 0.995)
        p_combined_full = torch.clamp(p_combined_full, 0.005, 0.995)
        p_tmle_loss = torch.clamp(p_tmle_loss, 0.005, 0.995)
        p_tmle_update = torch.clamp(p_tmle_update, 0.005, 0.995)
    else:
        print("Following Gruber et al. truncation")
        trunc_val = 5 / (np.sqrt(len(X_target)) * np.log(len(X_target)))
        p_std = torch.clamp(p_std, trunc_val, 1 - trunc_val)
        p_post_last = torch.clamp(p_post_last, trunc_val, 1 - trunc_val)
        p_post_full = torch.clamp(p_post_full, trunc_val, 1 - trunc_val)
        p_ttr_last = torch.clamp(p_ttr_last, trunc_val, 1 - trunc_val)
        p_ttr_full = torch.clamp(p_ttr_full, trunc_val, 1 - trunc_val)
        p_combined_last = torch.clamp(p_combined_last, trunc_val, 1 - trunc_val)
        p_combined_full = torch.clamp(p_combined_full, trunc_val, 1 - trunc_val)
        p_tmle_loss = torch.clamp(p_tmle_loss, trunc_val, 1 - trunc_val)
        p_tmle_update = torch.clamp(p_tmle_update, trunc_val, 1 - trunc_val)
    
    # Convert all predictions to numpy arrays
    p_std = p_std.cpu().detach().numpy().ravel()
    mu0_std = mu0_std.cpu().detach().numpy().ravel()
    mu1_std = mu1_std.cpu().detach().numpy().ravel()
    
    p_post_last = p_post_last.cpu().detach().numpy().ravel()
    mu0_post_last = mu0_post_last.cpu().detach().numpy().ravel()
    mu1_post_last = mu1_post_last.cpu().detach().numpy().ravel()
    
    p_post_full = p_post_full.cpu().detach().numpy().ravel()
    mu0_post_full = mu0_post_full.cpu().detach().numpy().ravel()
    mu1_post_full = mu1_post_full.cpu().detach().numpy().ravel()
    
    p_ttr_last = p_ttr_last.cpu().detach().numpy().ravel()
    mu0_ttr_last = mu0_ttr_last.cpu().detach().numpy().ravel()
    mu1_ttr_last = mu1_ttr_last.cpu().detach().numpy().ravel()
    
    p_ttr_full = p_ttr_full.cpu().detach().numpy().ravel()
    mu0_ttr_full = mu0_ttr_full.cpu().detach().numpy().ravel()
    mu1_ttr_full = mu1_ttr_full.cpu().detach().numpy().ravel()
    
    p_combined_last = p_combined_last.cpu().detach().numpy().ravel()
    mu0_combined_last = mu0_combined_last.cpu().detach().numpy().ravel()
    mu1_combined_last = mu1_combined_last.cpu().detach().numpy().ravel()
    
    p_combined_full = p_combined_full.cpu().detach().numpy().ravel()
    mu0_combined_full = mu0_combined_full.cpu().detach().numpy().ravel()
    mu1_combined_full = mu1_combined_full.cpu().detach().numpy().ravel()
    
    p_tmle_loss = p_tmle_loss.cpu().detach().numpy().ravel()
    mu0_tmle_loss = mu0_tmle_loss.cpu().detach().numpy().ravel()
    mu1_tmle_loss = mu1_tmle_loss.cpu().detach().numpy().ravel()
    
    p_tmle_update = p_tmle_update.cpu().detach().numpy().ravel()
    mu0_tmle_update = mu0_tmle_update.cpu().detach().numpy().ravel()
    mu1_tmle_update = mu1_tmle_update.cpu().detach().numpy().ravel()
    
    # Calculate model-based ATE (mean(u1-u0)) with confidence intervals for all models
    model_ate_std, model_se_std, model_cov_std, model_lstd, model_ustd, model_meanIFstd, model_varIFstd = model_ate_ci(
        Y_target, A_target, p_std, mu0_std, mu1_std, true_ate
    )
    
    model_ate_post_last, model_se_post_last, model_cov_post_last, model_lpost_last, model_upost_last, model_meanIFpost_last, model_varIFpost_last = model_ate_ci(
        Y_target, A_target, p_post_last, mu0_post_last, mu1_post_last, true_ate
    )
    
    model_ate_post_full, model_se_post_full, model_cov_post_full, model_lpost_full, model_upost_full, model_meanIFpost_full, model_varIFpost_full = model_ate_ci(
        Y_target, A_target, p_post_full, mu0_post_full, mu1_post_full, true_ate
    )
    
    model_ate_ttr_last, model_se_ttr_last, model_cov_ttr_last, model_lttr_last, model_uttr_last, model_meanIFttr_last, model_varIFttr_last = model_ate_ci(
        Y_target, A_target, p_ttr_last, mu0_ttr_last, mu1_ttr_last, true_ate
    )
    
    model_ate_ttr_full, model_se_ttr_full, model_cov_ttr_full, model_lttr_full, model_uttr_full, model_meanIFttr_full, model_varIFttr_full = model_ate_ci(
        Y_target, A_target, p_ttr_full, mu0_ttr_full, mu1_ttr_full, true_ate
    )
    
    model_ate_combined_last, model_se_combined_last, model_cov_combined_last, model_lcombined_last, model_ucombined_last, model_meanIFcombined_last, model_varIFcombined_last = model_ate_ci(
        Y_target, A_target, p_combined_last, mu0_combined_last, mu1_combined_last, true_ate
    )
    
    model_ate_combined_full, model_se_combined_full, model_cov_combined_full, model_lcombined_full, model_ucombined_full, model_meanIFcombined_full, model_varIFcombined_full = model_ate_ci(
        Y_target, A_target, p_combined_full, mu0_combined_full, mu1_combined_full, true_ate
    )
    
    model_ate_tmle_loss, model_se_tmle_loss, model_cov_tmle_loss, model_ltmle_loss, model_utmle_loss, model_meanIFtmle_loss, model_varIFtmle_loss = model_ate_ci(
        Y_target, A_target, p_tmle_loss, mu0_tmle_loss, mu1_tmle_loss, true_ate
    )
    
    model_ate_tmle_update, model_se_tmle_update, model_cov_tmle_update, model_ltmle_update, model_utmle_update, model_meanIFtmle_update, model_varIFtmle_update = model_ate_ci(
        Y_target, A_target, p_tmle_update, mu0_tmle_update, mu1_tmle_update, true_ate
    )
    
    # Calculate model-based (plug-in) with tmle-update for standard model
    model_ate_tmle, model_se_tmle, model_cov_tmle, model_meanIFtmle, model_varIFtmle = single_step_tmle(
        Y_target, A_target, p_std, mu0_std, mu1_std, true_ate
    )
    
    # Compute double-robust ATE estimates for standard model
    dr_ate_std, dr_se_std, dr_meanIFstd, dr_varIFstd = dr_ate(Y_target, A_target, p_std, mu0_std, mu1_std)
    dr_cov_std, dr_lstd, dr_ustd = coverage_and_ci(dr_ate_std, dr_se_std, true_ate)
    
    # Add all results to the result dictionary
    result_dict = {
        'true_ate': true_ate,
        
        # Standard model (Estimator 1)
        'model_ate_std': model_ate_std,
        'model_se_std': model_se_std,
        'model_cover_std': model_cov_std,
        'model_meanIF_std': model_meanIFstd,
        'model_varIF_std': model_varIFstd,
        
        # Post-training targeting, last-layer (Estimator 2)
        'model_ate_post_last': model_ate_post_last,
        'model_se_post_last': model_se_post_last,
        'model_cover_post_last': model_cov_post_last,
        'model_meanIF_post_last': model_meanIFpost_last,
        'model_varIF_post_last': model_varIFpost_last,
        
        # Post-training targeting, full-layers (Estimator 3)
        'model_ate_post_full': model_ate_post_full,
        'model_se_post_full': model_se_post_full,
        'model_cover_post_full': model_cov_post_full,
        'model_meanIF_post_full': model_meanIFpost_full,
        'model_varIF_post_full': model_varIFpost_full,
        
        # Training-targeting regularization, last-layer (Estimator 4)
        'model_ate_ttr_last': model_ate_ttr_last,
        'model_se_ttr_last': model_se_ttr_last,
        'model_cover_ttr_last': model_cov_ttr_last,
        'model_meanIF_ttr_last': model_meanIFttr_last,
        'model_varIF_ttr_last': model_varIFttr_last,
        
        # Training-targeting regularization, full-layers (Estimator 5)
        'model_ate_ttr_full': model_ate_ttr_full,
        'model_se_ttr_full': model_se_ttr_full,
        'model_cover_ttr_full': model_cov_ttr_full,
        'model_meanIF_ttr_full': model_meanIFttr_full,
        'model_varIF_ttr_full': model_varIFttr_full,
        
        # Combined approach: Training + Post-training targeting, last-layer (Estimator 6)
        'model_ate_combined_last': model_ate_combined_last,
        'model_se_combined_last': model_se_combined_last,
        'model_cover_combined_last': model_cov_combined_last,
        'model_meanIF_combined_last': model_meanIFcombined_last,
        'model_varIF_combined_last': model_varIFcombined_last,
        
        # Combined approach: Training + Post-training targeting, full-layers (Estimator 7)
        'model_ate_combined_full': model_ate_combined_full,
        'model_se_combined_full': model_se_combined_full,
        'model_cover_combined_full': model_cov_combined_full,
        'model_meanIF_combined_full': model_meanIFcombined_full,
        'model_varIF_combined_full': model_varIFcombined_full,
        
        # Training with TMLE-style loss (Estimator 8)
        'model_ate_tmle_loss': model_ate_tmle_loss,
        'model_se_tmle_loss': model_se_tmle_loss,
        'model_cover_tmle_loss': model_cov_tmle_loss,
        'model_meanIF_tmle_loss': model_meanIFtmle_loss,
        'model_varIF_tmle_loss': model_varIFtmle_loss,
        
        # Training with TMLE-style loss + TMLE update (Estimator 9)
        'model_ate_tmle_update': model_ate_tmle_update,
        'model_se_tmle_update': model_se_tmle_update,
        'model_cover_tmle_update': model_cov_tmle_update,
        'model_meanIF_tmle_update': model_meanIFtmle_update,
        'model_varIF_tmle_update': model_varIFtmle_update,
        'final_epsilon': epsilon_final,
        
        # Model-based with TMLE update (from standard model)
        'model_ate_tmle': model_ate_tmle,
        'model_se_tmle': model_se_tmle,
        'model_cover_tmle': model_cov_tmle,
        'model_meanIF_tmle': model_meanIFtmle,
        'model_varIF_tmle': model_varIFtmle,
        
        # DR AIPTW ATE (from standard model)
        'dr_ate_std': dr_ate_std,
        'dr_se_std': dr_se_std,
        'dr_cover_std': dr_cov_std,
        'dr_meanIF_std': dr_meanIFstd,
        'dr_varIF_std': dr_varIFstd,
        
        # Metrics
        'metrics_post_last': metrics_post_last,
        'metrics_post_full': metrics_post_full,
        'metrics_ttr_last': metrics_ttr_last,
        'metrics_ttr_full': metrics_ttr_full,
        'metrics_combined_last': metrics_combined_last,
        'metrics_combined_full': metrics_combined_full,
        'metrics_tmle_loss': metrics_tmle_loss,
        'metrics_tmle_update': metrics_tmle_update,
        
        # Additional information
        'train_size': len(X_train),
        'val_size': len(X_val),
        'target_size': len(X_target),
    }
    
    return result_dict