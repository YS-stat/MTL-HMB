# Standard libraries
import os
import time
import random
import itertools

# Third-party libraries
import numpy as np
import pandas as pd
import sklearn.model_selection
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from copy import deepcopy


def set_seed(seed):
    """Set seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data(X, y, train_ratio=0.6, val_ratio=0.2, seed=None):
    """Split data into training, validation, and testing sets with seed control."""
    total_size = len(X)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    dataset = TensorDataset(X, y)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return random_split(dataset, [train_size, val_size, test_size], generator=generator)


class Integmodel(nn.Module):
    """
    Integration model with shared and task-specific encoders for feature disentanglement.
    
    Args:
        input_dim: Dimension of shared input features
        input_dim_t: Dimension of target task-specific features
        input_dim_s: Dimension of source task-specific features
        shared_out_dim: Output dimension for shared encoder
        unique_out_dim: Output dimension for unique encoders
        depth: Number of layers in each submodule
        p1: Number of shared features
        p2: Number of target-specific features
        p3: Number of source-specific features
    """
    def __init__(self, input_dim, input_dim_t, input_dim_s, shared_out_dim, 
                 unique_out_dim, depth, p1, p2, p3, seed=None):
        super(Integmodel, self).__init__()
        
        # Set seed for reproducible weight initialization
        if seed is not None:
            torch.manual_seed(seed)
            
        self.depth = depth
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        def create_layers(in_features, out_features, depth):
            """Create sequential layers with linear and ReLU activation."""
            layers = []
            for _ in range(depth):
                layers.append(nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.ReLU()
                ))
                in_features = out_features
            return nn.Sequential(*layers)
        
        def create_layers_y(in_features, out_features, depth):
            """Create layers for shared feature processing with final prediction layer."""
            layers = []
            for _ in range(depth):
                layers.append(nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.ReLU()
                ))
                in_features = out_features
            layers.append(nn.Linear(out_features, 1))
            return nn.Sequential(*layers)
        
        def create_layers_y_unique(in_features, out_features, g_features, depth):
            """Create layers for task-specific feature processing with concatenation."""
            layers = []
            current_in = in_features
            for _ in range(depth):
                layers.append(nn.Sequential(
                    nn.Linear(current_in, out_features),
                    nn.ReLU()
                ))
                current_in = out_features + g_features
            layers.append(nn.Linear(current_in, 1))
            return nn.Sequential(*layers)

        # Shared encoder
        self.shared_encoder = create_layers(input_dim, shared_out_dim, depth)

        # Task-specific encoders
        self.unique_encoder_t = create_layers(input_dim_t, unique_out_dim, depth)
        self.unique_encoder_s = create_layers(input_dim_s, unique_out_dim, depth)

        # Feature processing paths
        self.g = create_layers_y(shared_out_dim + unique_out_dim, shared_out_dim, depth)
        self.g_t = create_layers_y_unique(shared_out_dim + unique_out_dim, 
                                         unique_out_dim, shared_out_dim, depth)
        self.g_s = create_layers_y_unique(shared_out_dim + unique_out_dim, 
                                         unique_out_dim, shared_out_dim, depth)
    
    def forward(self, x_t, x_s):
        """Forward pass with feature disentanglement and integration."""
        # Shared features
        f_t_c = self.shared_encoder(x_t[:, :self.p1])
        f_s_c = self.shared_encoder(x_s[:, :self.p1])

        # Task-specific features
        f_t_u = self.unique_encoder_t(x_t[:, self.p1:])
        f_s_u = self.unique_encoder_s(x_s[:, self.p1:])

        # Initial feature processing
        g_input_t = torch.cat([f_t_u, f_t_c], dim=1)
        g_input_s = torch.cat([f_s_u, f_s_c], dim=1)
        
        h_t = self.g_t[0](g_input_t)
        h_s = self.g_s[0](g_input_s)
        h_t_o = self.g[0](g_input_t)
        h_s_o = self.g[0](g_input_s)

        # Subsequent processing with regularization
        norm_products = 0
        for i in range(1, self.depth):
            # Concatenate outputs for next layer
            h_t_next = torch.cat([h_t, h_t_o], dim=1)
            h_s_next = torch.cat([h_s, h_s_o], dim=1)
            
            # Process features
            h_t = self.g_t[i](h_t_next)
            h_s = self.g_s[i](h_s_next)
            h_t_o = self.g[i](h_t_o)
            h_s_o = self.g[i](h_s_o)
            
            # Calculate regularization term
            half_size = h_t_o.size(1)
            wprod = (
                torch.matmul(self.g_s[i][0].weight[:, -half_size:], self.g[i][0].weight.t()) +
                torch.matmul(self.g_t[i][0].weight[:, -half_size:], self.g[i][0].weight.t())
            )
            norm_products += torch.norm(wprod) ** 2

        # Final predictions
        y_t = self.g_t[-1](torch.cat([h_t, h_t_o], dim=1)) + self.g[-1](h_t_o)
        y_s = self.g_s[-1](torch.cat([h_s, h_s_o], dim=1)) + self.g[-1](h_s_o)

        # Final regularization term
        half_size = h_t_o.size(1)
        wprod_last = (
            torch.matmul(self.g_s[-1].weight[:, -half_size:], self.g[-1].weight.t()) +
            torch.matmul(self.g_t[-1].weight[:, -half_size:], self.g[-1].weight.t())
        )
        norm_products += torch.norm(wprod_last) ** 2

        # Orthogonality regularization
        orth = torch.norm(f_t_c.t() @ f_t_u) ** 2 + torch.norm(f_s_c.t() @ f_s_u) ** 2

        return y_t, y_s, orth, norm_products


def grid_search_integmodel(X_t_hat, y_t, X_s_hat, y_s, p1, p2, p3, seed):
    """Perform grid search to find optimal model hyperparameters."""
    set_seed(seed)
    
    # Hyperparameter grid
    dims = [32, 64, 128]
    depths = [2, 3, 4]
    lrs = [0.001]
    lambda_orths = [1] 
    lambda_reds = [1]
    batch_size = 16 
    num_epochs = 25000
    patience = 30
    
    # Prepare data loaders with seed control
    train_set_t, val_set_t, test_set_t = prepare_data(X_t_hat, y_t, seed=seed)
    train_set_s, val_set_s, test_set_s = prepare_data(X_s_hat, y_s, seed=seed)

    train_loader_t = DataLoader(train_set_t, batch_size=batch_size, shuffle=True)
    val_loader_t = DataLoader(val_set_t, batch_size=batch_size, shuffle=False)
    test_loader_t = DataLoader(test_set_t, batch_size=batch_size, shuffle=False)

    train_loader_s = DataLoader(train_set_s, batch_size=batch_size, shuffle=True)
    val_loader_s = DataLoader(val_set_s, batch_size=batch_size, shuffle=False)
    test_loader_s = DataLoader(test_set_s, batch_size=batch_size, shuffle=False)

    best_overall_val_loss = float('inf')
    best_model = None
    best_params = None

    # Grid search over hyperparameters
    for dim, depth, lr, lambda_orth, lambda_red in itertools.product(
        dims, depths, lrs, lambda_orths, lambda_reds
    ):
        # Initialize model with seed for reproducible weights
        model = Integmodel(
            input_dim=p1, 
            input_dim_t=p2,
            input_dim_s=p3,
            shared_out_dim=dim,
            unique_out_dim=dim,
            depth=depth,
            p1=p1,
            p2=p2,
            p3=p3,
            seed=seed
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)
        
        best_val_loss = float('inf')
        num_bad_epochs = 0
        current_best_model = None

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for (batch_t, batch_s) in zip(train_loader_t, train_loader_s):
                X_t, y_t_batch = batch_t
                X_s, y_s_batch = batch_s
                
                optimizer.zero_grad()
                y_t_pred, y_s_pred, orth, red = model(X_t, X_s)
                
                # Calculate losses
                loss_t = criterion(y_t_pred, y_t_batch)
                loss_s = criterion(y_s_pred, y_s_batch)
                total_loss = loss_t + loss_s + lambda_orth * orth + lambda_red * red
                
                total_loss.backward()
                optimizer.step()
            
            scheduler.step()

            # Validation
            model.eval()
            total_val_loss = 0
            val_batches = 0
            
            for (batch_t, batch_s) in zip(val_loader_t, val_loader_s):
                X_t, y_t_batch = batch_t
                X_s, y_s_batch = batch_s
                
                with torch.no_grad():
                    y_t_pred, y_s_pred, _, _ = model(X_t, X_s)
                    loss_t = criterion(y_t_pred, y_t_batch)
                    loss_s = criterion(y_s_pred, y_s_batch)
                    total_val_loss += loss_t.item() + loss_s.item()
                    val_batches += 1
            
            avg_val_loss = total_val_loss / val_batches
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                current_best_model = deepcopy(model.state_dict())
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
                if num_bad_epochs >= patience:
                    break

        # Update overall best model
        if best_val_loss < best_overall_val_loss:
            best_overall_val_loss = best_val_loss
            best_params = (dim, depth, lr, lambda_orth, lambda_red)
            best_model = current_best_model

    # Load best model for testing
    model = Integmodel(
        input_dim=p1, 
        input_dim_t=p2,
        input_dim_s=p3,
        shared_out_dim=best_params[0],
        unique_out_dim=best_params[0],
        depth=best_params[1],
        p1=p1,
        p2=p2,
        p3=p3,
        seed=seed
    )
    model.load_state_dict(best_model)
    model.eval()
    
    # Evaluate on test set
    criterion = nn.MSELoss()
    test_loss_t, test_loss_s = 0.0, 0.0
    rel_error_t, rel_error_s = 0.0, 0.0
    test_batches = 0

    for (batch_t, batch_s) in zip(test_loader_t, test_loader_s):
        X_t, y_t_batch = batch_t
        X_s, y_s_batch = batch_s
        
        with torch.no_grad():
            y_t_pred, y_s_pred, _, _ = model(X_t, X_s)
            
            # RMSE calculation
            test_loss_t += torch.sqrt(criterion(y_t_pred, y_t_batch)).item()
            test_loss_s += torch.sqrt(criterion(y_s_pred, y_s_batch)).item()
            
            # Relative error calculation
            null_pred_error_t = criterion(torch.zeros_like(y_t_batch), y_t_batch)
            null_pred_error_s = criterion(torch.zeros_like(y_s_batch), y_s_batch)
            rel_error_t += criterion(y_t_pred, y_t_batch) / null_pred_error_t
            rel_error_s += criterion(y_s_pred, y_s_batch) / null_pred_error_s
            
            test_batches += 1
    
    # Calculate averages
    avg_test_loss_t = test_loss_t / test_batches
    avg_test_loss_s = test_loss_s / test_batches
    avg_rel_error_t = rel_error_t / test_batches
    avg_rel_error_s = rel_error_s / test_batches
    
    return avg_test_loss_s, avg_test_loss_t, avg_rel_error_s, avg_rel_error_t


def run_simulation_HTL(p1, p2, p3, rho1, rho2, n_t, n_s, alpha, sigmat, sigmas, num_seeds, output_dir):
    """
    Run simulation experiments for HTL method.
    
    Args:
        p1: Number of shared features
        p2: Number of target-specific features
        p3: Number of source-specific features
        rho1: Correlation parameter for target data
        rho2: Correlation parameter for source data
        n_t: Target sample size
        n_s: Source sample size
        alpha: Mixing parameter for response generation
        sigmat: Noise level for target response
        sigmas: Noise level for source response
        num_seeds: Number of random seeds to test
        output_dir: Directory to save results
    """
    results = []
    p_total = p1 + p2 + p3
    
    for seed in tqdm(range(num_seeds), desc="Processing Seeds"):
        # Set all seeds for reproducibility
        set_seed(seed)
        
        # Generate multivariate normal features
        p = p1 + p2 + p3
        def generate_correlated_data(rho, n_samples):
            cov = torch.zeros(p, p)
            for i in range(p):
                for j in range(p):
                    cov[i, j] = rho ** (0.01 * abs(i - j))
            cov.fill_diagonal_(1)
            dist = torch.distributions.MultivariateNormal(torch.zeros(p), covariance_matrix=cov)
            return dist.sample((n_samples,))

        X_t = generate_correlated_data(rho1, n_t)
        X_s = generate_correlated_data(rho2, n_s)

        X_t_1, X_t_2, X_t_3 = X_t[:, :p1], X_t[:, p1:p1+p2], X_t[:, p1+p2:]
        X_s_1, X_s_2, X_s_3 = X_s[:, :p1], X_s[:, p1:p1+p2], X_s[:, p1+p2:]

        # Generate response variables
        mean, std = -10, 10
        v_c = torch.randn(p) * std + mean
        v_t = torch.randn(p) * std + mean
        v_s = torch.randn(p) * std + mean
        epsilon_t = torch.randn(n_t) * sigmat
        epsilon_s = torch.randn(n_s) * sigmas

        y_t = alpha * torch.sum((X_t ** 2) * v_c, dim=1) / p + (1 - alpha) * torch.sum(X_t * v_t, dim=1) / p + epsilon_t
        y_s = alpha * torch.sum((X_s ** 2) * v_c, dim=1) / p + (1 - alpha) * torch.sum(X_s * v_s, dim=1) / p + epsilon_s
        y_t, y_s = y_t.view(-1, 1), y_s.view(-1, 1)
                
        start_time = time.time()
        
        # Prepare input features
        X_t_hat = torch.cat([X_t_1, X_t_2], dim=1)
        X_s_hat = torch.cat([X_s_1, X_s_3], dim=1)
        
        # Run grid search and evaluation
        loss_s, loss_t, rel_err_s, rel_err_t = grid_search_integmodel(
            X_t_hat, y_t, X_s_hat, y_s, p1, p2, p3, seed=seed
        )
        
        elapsed_time = time.time() - start_time
        results.append((seed, loss_s, loss_t, rel_err_s, rel_err_t, elapsed_time))
        
        # Print progress
        # print(f"Seed {seed}: Loss_s = {loss_s:.4f}, Loss_t = {loss_t:.4f}, "
        #       f"Rel_s = {rel_err_s:.4f}, Rel_t = {rel_err_t:.4f}, "
        #       f"Time = {elapsed_time:.2f}s")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results, columns=[
        'Seed', 'Loss_s', 'Loss_t', 'Rel_Error_s', 'Rel_Error_t', 'Time'
    ])
    
    filename = (
        f"HTL_p1={p1}_p2={p2}_p3={p3}_rho1={rho1}_rho2={rho2}_"
        f"nt={n_t}_ns={n_s}_alpha={alpha}_sigmat={sigmat}_"
        f"sigmas={sigmas}_seeds={num_seeds}.csv"
    )
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    
    print(f"HTL Results saved to {filepath}")