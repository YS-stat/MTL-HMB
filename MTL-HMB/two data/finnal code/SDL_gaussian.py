# Standard libraries
import os
import time
import random
import itertools

# Third-party libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from copy import deepcopy


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data(X, y, train_ratio=0.6, val_ratio=0.2, seed=None):
    """Split data into training, validation and test sets with seed control."""
    total_size = len(X)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    dataset = TensorDataset(X, y)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return random_split(dataset, [train_size, val_size, test_size], generator=generator)


class FlexibleMLP(nn.Module):
    """Configurable Multi-Layer Perceptron with variable depth and width."""
    
    def __init__(self, input_dim, depth, width):
        super(FlexibleMLP, self).__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Sequential(
                nn.Linear(input_dim, width),
                nn.ReLU()
            ))
            input_dim = width
        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def grid_search_sdl(X, y, input_dim, seed):
    """Perform hyperparameter grid search for SDL model."""
    set_seed(seed)
    
    # Hyperparameter grid
    dims = [32, 64, 128]
    depths = [2, 3, 4]
    lrs = [0.001]
    batch_size = 16
    num_epochs = 25000
    patience = 30
    
    # Prepare data loaders with seed control
    train_set, val_set, test_set = prepare_data(X, y, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_model = None
    best_params = None

    # Grid search over hyperparameters
    for dim, depth, lr in itertools.product(dims, depths, lrs):
        model = FlexibleMLP(input_dim, depth, dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)
        
        current_best_loss = float('inf')
        bad_epochs = 0
        current_best_model = None

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    y_pred = model(X_val)
                    val_loss += criterion(y_pred, y_val).item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Early stopping check
            if avg_val_loss < current_best_loss:
                current_best_loss = avg_val_loss
                current_best_model = deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        # Update overall best model
        if current_best_loss < best_val_loss:
            best_val_loss = current_best_loss
            best_params = (dim, depth, lr)
            best_model = current_best_model

    # Load best model for testing
    model = FlexibleMLP(input_dim, best_params[1], best_params[0])
    model.load_state_dict(best_model)
    model.eval()
    
    # Evaluation metrics
    criterion = nn.MSELoss()
    test_loss = 0
    rel_error = 0
    
    with torch.no_grad():
        for X_test, y_test in test_loader:
            y_pred = model(X_test)
            test_loss += torch.sqrt(criterion(y_pred, y_test)).item()
            null_error = criterion(torch.zeros_like(y_test), y_test)
            rel_error += criterion(y_pred, y_test) / null_error
    
    avg_test_loss = test_loss / len(test_loader)
    avg_rel_error = rel_error / len(test_loader)
    
    return avg_test_loss, avg_rel_error


def run_simulation_SDL(p1, p2, p3, rho1, rho2, n_t, n_s, alpha, sigmat, sigmas, num_seeds, output_dir):
    """Run simulation experiments for SDL method.""" 
    results = []
    
    for seed in tqdm(range(num_seeds), desc="Processing Seeds"):
        # Set seeds for reproducibility
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
        
        # Train and evaluate models
        X_t_hat = torch.cat([X_t_1, X_t_2], dim=1)
        loss_t, rel_err_t = grid_search_sdl(X_t_hat, y_t, p1+p2, seed)
        
        X_s_hat = torch.cat([X_s_1, X_s_3], dim=1)
        loss_s, rel_err_s = grid_search_sdl(X_s_hat, y_s, p1+p3, seed)
        
        elapsed_time = time.time() - start_time
        
        # Store results
        results.append((seed, loss_s, loss_t, rel_err_s, rel_err_t, elapsed_time))
        # print(f"Seed {seed}: Loss_s = {loss_s:.4f}, Loss_t = {loss_t:.4f}, "
        #       f"Rel_s = {rel_err_s:.4f}, Rel_t = {rel_err_t:.4f}, "
        #       f"Time = {elapsed_time:.2f}s")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results, columns=[
        'Seed', 'Loss_s', 'Loss_t', 'Rel_Error_s', 'Rel_Error_t', 'Time'
    ])
    
    filename = (f"SDL_p1={p1}_p2={p2}_p3={p3}_rho1={rho1}_rho2={rho2}_"
               f"nt={n_t}_ns={n_s}_alpha={alpha}_sigmat={sigmat}_"
               f"sigmas={sigmas}_seeds={num_seeds}.csv")
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    
    print(f"SDL Results saved to {filepath}")