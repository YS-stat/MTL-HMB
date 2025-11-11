# Standard libraries
import os
import time
import random
import itertools

# Third-party libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from sklearn.model_selection import train_test_split

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split


def set_seed(seed):
    """Set seeds for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # 添加确定性计算标志

def prepare_data(X, y, train_ratio=0.6, val_ratio=0.2, seed=None):
    """Split data into training, validation, and testing sets with seed control."""
    total_size = len(X)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    dataset = TensorDataset(X, y)
    
    # 使用种子控制的生成器
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return random_split(dataset, [train_size, val_size, test_size], generator=generator)


class FlexibleMLP(nn.Module):
    """Flexible multi-layer perceptron with configurable depth and width."""
    def __init__(self, input_dim, output_dim, depth, width):
        super(FlexibleMLP, self).__init__()
        layers = []
        for i in range(depth):
            if i < depth - 1:
                layers.append(nn.Sequential(nn.Linear(input_dim, width), nn.ReLU()))
                input_dim = width
            else:
                layers.append(nn.Linear(input_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def imputation_grid_search(X_t_1_train, X_t_1_val, X_s_1_train, X_s_1_val,
                           X_t_2_train, X_t_2_val, X_s_1, input_dim, output_dim, seed):
    """
    Grid search over neural network hyperparameters to impute missing features.
    Shared encoder + unique encoders + decoder + predictor.
    """
    set_seed(seed)
    widths = [8, 16, 32]
    depths = [1, 2, 3]
    lrs = [0.001]
    lambda_orths = [0, 1]

    best_eval_loss = float("inf")
    best_model = None
    best_config = None

    for width, depth, lr, lambda_orth in itertools.product(widths, depths, lrs, lambda_orths):
        shared_encoder = FlexibleMLP(input_dim, width, depth, width)
        unique_encoder_t = FlexibleMLP(input_dim, width, depth, width)
        unique_encoder_s = FlexibleMLP(input_dim, width, depth, width)
        decoder = FlexibleMLP(2 * width, input_dim, depth, input_dim)
        predictor = FlexibleMLP(width, output_dim, depth, width)

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(
            list(shared_encoder.parameters()) +
            list(unique_encoder_t.parameters()) +
            list(unique_encoder_s.parameters()) +
            list(decoder.parameters()) +
            list(predictor.parameters()),
            lr=lr
        )

        num_epochs = 5000
        patience = 30
        num_bad_epochs = 0

        for epoch in range(num_epochs):
            # Training
            optimizer.zero_grad()
            shared_encoder.train()
            unique_encoder_t.train()
            unique_encoder_s.train()
            decoder.train()
            predictor.train()

            f_t_c = shared_encoder(X_t_1_train)
            f_s_c = shared_encoder(X_s_1_train)
            f_t_u = unique_encoder_t(X_t_1_train)
            f_s_u = unique_encoder_s(X_s_1_train)

            X_t_1_hat = decoder(torch.cat([f_t_c, f_t_u], dim=1))
            X_s_1_hat = decoder(torch.cat([f_s_c, f_s_u], dim=1))
            X_t_2_hat = predictor(f_t_c)

            loss_X_t_1 = loss_fn(X_t_1_hat, X_t_1_train)
            loss_X_s_1 = loss_fn(X_s_1_hat, X_s_1_train)
            loss_X_t_2 = loss_fn(X_t_2_hat, X_t_2_train)

            orth = torch.norm(f_t_c.t() @ f_t_u) + torch.norm(f_s_c.t() @ f_s_u)
            total_loss = loss_X_t_1 + loss_X_s_1 + loss_X_t_2 + lambda_orth * orth
            total_loss.backward()
            optimizer.step()

            # Validation
            shared_encoder.eval()
            with torch.no_grad():
                f_t_c_val = shared_encoder(X_t_1_val)
                X_t_2_hat_val = predictor(f_t_c_val)
                eval_loss = loss_fn(X_t_2_hat_val, X_t_2_val)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_config = [width, depth, lr, lambda_orth]
                best_model = {
                    'shared_encoder': deepcopy(shared_encoder),
                    'predictor': deepcopy(predictor)
                }
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1

            if num_bad_epochs >= patience:
                break

    #print("Best config:", best_config)

    # Final prediction
    shared_encoder = best_model['shared_encoder']
    predictor = best_model['predictor']
    shared_encoder.eval()
    predictor.eval()
    with torch.no_grad():
        f_s_c = shared_encoder(X_s_1)
        X_s_2_hat_best = predictor(f_s_c)

    return X_s_2_hat_best


class Integmodel(nn.Module):
    """
    Integration model with shared and task-specific encoders.
    Applies layered transformations and regularizations to enforce
    feature disentanglement and structural alignment.
    """
    def __init__(self, input_dim, input_dim_t, input_dim_s, shared_out_dim, unique_out_dim, depth, p1, p2, p3, seed=None):
        super(Integmodel, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.depth = depth
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        def create_layers(in_features, out_features, depth):
            layers = []
            for _ in range(depth):
                layers.append(nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU()))
                in_features = out_features
            return nn.Sequential(*layers)

        def create_layers_y(in_features, out_features, depth):
            layers = []
            for _ in range(depth):
                layers.append(nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU()))
                in_features = out_features
            layers.append(nn.Linear(in_features, 1))  # final prediction layer
            return nn.Sequential(*layers)

        def create_layers_y_unique(in_features, out_features, g_features, depth):
            layers = []
            current_in = in_features
            for _ in range(depth):
                layers.append(nn.Sequential(nn.Linear(current_in, out_features), nn.ReLU()))
                current_in = out_features + g_features
            layers.append(nn.Linear(current_in, 1))
            return nn.Sequential(*layers)

        self.shared_encoder = create_layers(input_dim, shared_out_dim, depth)
        self.unique_encoder_t = create_layers(input_dim_t, unique_out_dim, depth)
        self.unique_encoder_s = create_layers(input_dim_s, unique_out_dim, depth)
        self.g = create_layers_y(shared_out_dim + unique_out_dim, shared_out_dim, depth)
        self.g_t = create_layers_y_unique(shared_out_dim + unique_out_dim, unique_out_dim, shared_out_dim, depth)
        self.g_s = create_layers_y_unique(shared_out_dim + unique_out_dim, unique_out_dim, shared_out_dim, depth)

    def forward(self, x_t, x_s):
        # Shared features
        f_t_c = self.shared_encoder(x_t[:, :self.p1])
        f_s_c = self.shared_encoder(x_s[:, :self.p1])

        # Task-specific features
        f_t_u = self.unique_encoder_t(x_t[:, self.p1:])
        f_s_u = self.unique_encoder_s(x_s[:, self.p1:])

        g_input_t = torch.cat([f_t_u, f_t_c], dim=1)
        g_input_s = torch.cat([f_s_u, f_s_c], dim=1)

        h_t = self.g_t[0](g_input_t)
        h_s = self.g_s[0](g_input_s)
        h_t_o = self.g[0](g_input_t)
        h_s_o = self.g[0](g_input_s)

        norm_products = 0
        for i in range(1, self.depth):
            h_t = self.g_t[i](torch.cat([h_t, h_t_o], dim=1))
            h_s = self.g_s[i](torch.cat([h_s, h_s_o], dim=1))
            h_t_o = self.g[i](h_t_o)
            h_s_o = self.g[i](h_s_o)

            half_size = h_t_o.size(1)
            wprod = (
                torch.matmul(self.g_s[i][0].weight[:, -half_size:], self.g[i][0].weight.t()) +
                torch.matmul(self.g_t[i][0].weight[:, -half_size:], self.g[i][0].weight.t())
            )
            norm_products += torch.norm(wprod) ** 2

        # Final outputs
        y_t = self.g_t[-1](torch.cat([h_t, h_t_o], dim=1)) + self.g[-1](h_t_o)
        y_s = self.g_s[-1](torch.cat([h_s, h_s_o], dim=1)) + self.g[-1](h_s_o)

        # Final norm regularization
        half_size = h_t_o.size(1)
        wprod_last = (
            torch.matmul(self.g_s[-1].weight[:, -half_size:], self.g[-1].weight.t()) +
            torch.matmul(self.g_t[-1].weight[:, -half_size:], self.g[-1].weight.t())
        )
        norm_products += torch.norm(wprod_last) ** 2

        # Orthogonality regularization
        orth = torch.norm(f_t_c.t() @ f_t_u) ** 2 + torch.norm(f_s_c.t() @ f_s_u) ** 2

        # Robustness penalty (on tail blocks of unique encoders)
        rob_penalty = (
            torch.sum(self.unique_encoder_t[0][0].weight[:, self.p1 + self.p2:] ** 2) +
            torch.sum(self.unique_encoder_s[0][0].weight[:, self.p1:self.p1 + self.p2] ** 2)
        )

        return y_t, y_s, orth, rob_penalty, norm_products


def grid_search_integmodel(X_t_hat, y_t, X_s_hat, y_s, p1, p2, p3, seed):
    """
    Perform grid search to train Integmodel with regularization.
    Evaluate performance on validation and test sets.
    """
    set_seed(seed)

    # Hyperparameter grid
    dims = [32, 64, 128]
    depths = [2, 3, 4]
    lrs = [0.001]
    lambda_orths = [0.01, 0.1]
    lambda_robs = [0.01, 0.1]
    lambda_reds = [0.01, 0.1]
    batch_size = 16
    num_epochs = 25000

    input_dim = p1
    input_dim_t = p1 + p2 + p3
    input_dim_s = p1 + p2 + p3

    # Data split
    train_set_t, val_set_t, test_set_t = prepare_data(X_t_hat, y_t)
    train_set_s, val_set_s, test_set_s = prepare_data(X_s_hat, y_s)

    train_loader_t = DataLoader(train_set_t, batch_size=batch_size, shuffle=True)
    val_loader_t = DataLoader(val_set_t, batch_size=batch_size, shuffle=False)
    test_loader_t = DataLoader(test_set_t, batch_size=batch_size, shuffle=False)

    train_loader_s = DataLoader(train_set_s, batch_size=batch_size, shuffle=True)
    val_loader_s = DataLoader(val_set_s, batch_size=batch_size, shuffle=False)
    test_loader_s = DataLoader(test_set_s, batch_size=batch_size, shuffle=False)

    best_overall_val_loss = float("inf")
    best_model = None
    best_paras = None

    # Grid search loop
    for dim, depth, lr, lambda_orth, lambda_rob, lambda_red in itertools.product(
        dims, depths, lrs, lambda_orths, lambda_robs, lambda_reds
    ):
        model = Integmodel(input_dim, input_dim_t, input_dim_s,
                           shared_out_dim=dim, unique_out_dim=dim,
                           depth=depth, p1=p1, p2=p2, p3=p3)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)

        best_val_loss = float("inf")
        num_bad_epochs = 0
        patience = 30

        for epoch in range(num_epochs):
            model.train()
            for (batch_t, batch_s) in zip(train_loader_t, train_loader_s):
                X_t, y_t_batch = batch_t
                X_s, y_s_batch = batch_s
                optimizer.zero_grad()
                y_t_pred, y_s_pred, orth, rob, red = model(X_t, X_s)
                loss_t = criterion(y_t_pred, y_t_batch)
                loss_s = criterion(y_s_pred, y_s_batch)
                total_loss = loss_t + loss_s + lambda_orth * orth + lambda_rob * rob + lambda_red * red
                total_loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation
            model.eval()
            total_val_loss = 0
            val_batch = 0
            for (batch_t, batch_s) in zip(val_loader_t, val_loader_s):
                X_t, y_t_batch = batch_t
                X_s, y_s_batch = batch_s
                with torch.no_grad():
                    y_t_pred, y_s_pred, _, _, _ = model(X_t, X_s)
                    loss_t = criterion(y_t_pred, y_t_batch)
                    loss_s = criterion(y_s_pred, y_s_batch)
                    total_val_loss += loss_t.item() + loss_s.item()
                    val_batch += 1

            avg_val_loss = total_val_loss / val_batch
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = deepcopy(model)
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
                if num_bad_epochs >= patience:
                    break

        # Update best model across all settings
        if best_val_loss < best_overall_val_loss:
            best_overall_val_loss = best_val_loss
            best_paras = [dim, depth, lr, lambda_orth, lambda_rob, lambda_red]
            best_model_settings = best_model

    # Evaluation on test set
    print("Best config:", best_paras)
    model = best_model_settings
    model.eval()
    criterion = nn.MSELoss()
    test_loss_t = 0.0
    test_loss_s = 0.0
    rel_error_t = 0.0
    rel_error_s = 0.0
    test_batch = 0

    for (batch_t, batch_s) in zip(test_loader_t, test_loader_s):
        X_t, y_t_batch = batch_t
        X_s, y_s_batch = batch_s
        with torch.no_grad():
            y_t_pred, y_s_pred, _, _, _ = model(X_t, X_s)
            test_loss_t += torch.sqrt(criterion(y_t_pred, y_t_batch)).item()
            test_loss_s += torch.sqrt(criterion(y_s_pred, y_s_batch)).item()
            rel_error_t += (criterion(y_t_pred, y_t_batch) / criterion(torch.zeros_like(y_t_batch), y_t_batch)).item()
            rel_error_s += (criterion(y_s_pred, y_s_batch) / criterion(torch.zeros_like(y_s_batch), y_s_batch)).item()
            test_batch += 1

    avg_test_loss_s = test_loss_s / test_batch
    avg_test_loss_t = test_loss_t / test_batch
    avg_rel_error_s = rel_error_s / test_batch
    avg_rel_error_t = rel_error_t / test_batch

    return avg_test_loss_s, avg_test_loss_t, avg_rel_error_s, avg_rel_error_t


def run_simulation_proposed(p1, p2, p3, rho1, rho2, n_t, n_s, alpha, sigmat, sigmas, num_seeds, output_dir):
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

        # Impute X_s_2 using shared encoder trained on (X_t_1, X_t_2)
        X_t_1_train, X_t_1_val = train_test_split(X_t_1, test_size=0.1, random_state=seed)
        X_s_1_train, X_s_1_val = train_test_split(X_s_1, test_size=0.1, random_state=seed)
        X_t_2_train, X_t_2_val = train_test_split(X_t_2, test_size=0.1, random_state=seed)

        X_s_2_hat = imputation_grid_search(X_t_1_train, X_t_1_val, X_s_1_train, X_s_1_val,
                                           X_t_2_train, X_t_2_val, X_s_1,
                                           input_dim=p1, output_dim=p2, seed=seed)

        #print(f"[Seed {seed}] Imputation error X_s_2: {torch.norm(X_s_2_hat - X_s_2) / torch.norm(X_s_2):.4f}")

        # Impute X_t_3 using shared encoder trained on (X_s_1, X_s_3)
        X_s_3_train, X_s_3_val = train_test_split(X_s_3, test_size=0.1, random_state=seed)
        X_t_3_hat = imputation_grid_search(X_s_1_train, X_s_1_val, X_t_1_train, X_t_1_val,
                                           X_s_3_train, X_s_3_val, X_t_1,
                                           input_dim=p1, output_dim=p3, seed=seed)

        #print(f"[Seed {seed}] Imputation error X_t_3: {torch.norm(X_t_3_hat - X_t_3) / torch.norm(X_t_3):.4f}")

        # Construct input for integration model
        X_t_hat = torch.cat([X_t_1, X_t_1, X_t_2, X_t_3_hat], dim=1)
        X_s_hat = torch.cat([X_s_1, X_s_1, X_s_2_hat, X_s_3], dim=1)

        # Run integrative training and evaluation
        loss_s, loss_t, rel_err_s, rel_err_t = grid_search_integmodel(
            X_t_hat, y_t, X_s_hat, y_s, p1, p2, p3, seed=seed
        )

        elapsed_time = time.time() - start_time
        #print(f"[Seed {seed}] Loss_s = {loss_s:.4f}, Loss_t = {loss_t:.4f}, Rel_s = {rel_err_s:.4f}, Rel_t = {rel_err_t:.4f}, Time = {elapsed_time:.2f}s")

        results.append((seed, loss_s, loss_t, rel_err_s, rel_err_t, elapsed_time))

    # Save results to CSV
    df = pd.DataFrame(results, columns=["Seed", "Loss_s", "Loss_t", "Rel_Error_s", "Rel_Error_t", "Time"])
    file_name = f"new_proposed_p1={p1}_p2={p2}_p3={p3}_rho1={rho1}_rho2={rho2}_nt={n_t}_ns={n_s}_alpha={alpha}_sigmat={sigmat}_sigmas={sigmas}_seeds={num_seeds}.csv"
    file_path = os.path.join(output_dir, file_name)
    df.to_csv(file_path, index=False)
    print(f"Proposed Results saved to {file_path}")

