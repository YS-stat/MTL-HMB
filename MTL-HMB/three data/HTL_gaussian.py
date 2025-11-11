
import os
import numpy as np
import pandas as pd
import random
import time
import sklearn.model_selection
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import itertools  
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
    def __init__(self, input_dim, input_dim_t, input_dim_s, input_dim_r, shared_out_dim, unique_out_dim, depth):
        super(Integmodel, self).__init__()
        self.depth = depth

        def create_layers(in_features, out_features, depth):
            layers = []
            for _ in range(depth):
                layers.append(nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.ReLU()
                ))
                in_features = out_features
            return nn.Sequential(*layers)
        
        def create_layers_y_unique(in_features, out_features, g_features, depth):
            layers = []
            current_in_features = in_features
            for _ in range(depth):
                layers.append(nn.Sequential(
                    nn.Linear(current_in_features, out_features),
                    nn.ReLU()
                ))
                current_in_features = out_features + g_features
            layers.append(nn.Linear(current_in_features, 1))  # Ensure the final layer outputs dimension 1
            return nn.Sequential(*layers)
        
        def create_layers_y(in_features, out_features, depth):
            layers = []
            current_in_features = in_features
            for _ in range(depth):
                layers.append(nn.Sequential(
                    nn.Linear(current_in_features, out_features),
                    nn.ReLU()
                ))
                current_in_features = out_features
            layers.append(nn.Linear(current_in_features, 1))  # Ensure the final layer outputs dimension 1
            return nn.Sequential(*layers)

        self.shared_encoder = create_layers(input_dim, shared_out_dim, depth)

        self.unique_encoder_t = create_layers(input_dim_t, unique_out_dim, depth)

        self.unique_encoder_s = create_layers(input_dim_s, unique_out_dim, depth)
        
        self.unique_encoder_r = create_layers(input_dim_r, unique_out_dim, depth)

        self.g = create_layers_y(shared_out_dim + unique_out_dim, shared_out_dim, depth)
        
        self.g_t = create_layers_y_unique(shared_out_dim + unique_out_dim, unique_out_dim, shared_out_dim, depth)

        self.g_s = create_layers_y_unique(shared_out_dim + unique_out_dim, unique_out_dim, shared_out_dim, depth)
        
        self.g_r = create_layers_y_unique(shared_out_dim + unique_out_dim, unique_out_dim, shared_out_dim, depth)
    
    def forward(self, x_t, x_s, x_r):
        # common encoding
        f_t_c = self.shared_encoder(x_t[:, :p1])
        f_s_c = self.shared_encoder(x_s[:, :p1])
        f_r_c = self.shared_encoder(x_r[:, :p1])

        # unique encoding
        f_t_u = self.unique_encoder_t(x_t[:, p1:])
        f_s_u = self.unique_encoder_s(x_s[:, p1:])
        f_r_u = self.unique_encoder_r(x_r[:, p1:])

        g_input_s = torch.cat([f_s_u, f_s_c], dim=1)
        g_input_t = torch.cat([f_t_u, f_t_c], dim=1)
        g_input_r = torch.cat([f_r_u, f_r_c], dim=1)

        h_t = self.g_t[0](g_input_t)
        h_s = self.g_s[0](g_input_s)
        h_r = self.g_r[0](g_input_r)
        h_t_o = self.g[0](g_input_t)
        h_s_o = self.g[0](g_input_s)
        h_r_o = self.g[0](g_input_r)

        norm_products = 0
        for i in range(1, self.depth):
            h_t_next_input = torch.cat([h_t, h_t_o], dim=1)  
            h_s_next_input = torch.cat([h_s, h_s_o], dim=1)
            h_r_next_input = torch.cat([h_r, h_r_o], dim=1)
            
            h_t = self.g_t[i](h_t_next_input) 
            h_s = self.g_s[i](h_s_next_input)
            h_r = self.g_r[i](h_r_next_input)
            h_t_o = self.g[i](h_t_o) 
            h_s_o = self.g[i](h_s_o)
            h_r_o = self.g[i](h_r_o)
            
            half_input_size = h_t_o.size(1)
            weight_product = torch.matmul(self.g_s[i][0].weight[:, -half_input_size:], self.g[i][0].weight.t()) +                              torch.matmul(self.g_t[i][0].weight[:, -half_input_size:], self.g[i][0].weight.t()) +                              torch.matmul(self.g_r[i][0].weight[:, -half_input_size:], self.g[i][0].weight.t())

            norm_products += (torch.norm(weight_product))**2

        y_s= self.g_s[-1](torch.cat([h_s, h_s_o], dim=1))+self.g[-1](h_s_o)
        y_t= self.g_t[-1](torch.cat([h_t, h_t_o], dim=1))+self.g[-1](h_t_o)
        y_r= self.g_r[-1](torch.cat([h_r, h_r_o], dim=1))+self.g[-1](h_r_o)
        
        half_input_size = h_t_o.size(1)
        weight_product = torch.matmul(self.g_s[-1].weight[:, -half_input_size:], self.g[-1].weight.t()) + torch.matmul(self.g_t[-1].weight[:, -half_input_size:], self.g[-1].weight.t()) + torch.matmul(self.g_r[-1].weight[:, -half_input_size:], self.g[-1].weight.t())
        norm_products += (torch.norm(weight_product))**2
        orth = torch.norm(f_t_c.t() @ f_t_u)**2 + torch.norm(f_s_c.t() @ f_s_u)**2 + torch.norm(f_r_c.t() @ f_r_u)**2
    
        return y_t, y_s, y_r, orth, norm_products


def grid_search_integmodel(X_t_hat, y_t, X_s_hat, y_s, X_r_hat, y_r, seed):
    set_seed(seed)
    dims = [32, 64, 128]
    depths = [2,3,4]
    lrs = [0.001]
    lambda_orths =[1] 
    lambda_reds = [1]
    batch_size = 16 
    num_epochs = 25000
    input_dim=p1
    input_dim_t=p2
    input_dim_s=p3
    input_dim_r=p4
    
    train_set_t, val_set_t, test_set_t = prepare_data(X_t_hat, y_t, seed=seed) 
    train_set_s, val_set_s, test_set_s = prepare_data(X_s_hat, y_s, seed=seed)
    train_set_r, val_set_r, test_set_r = prepare_data(X_r_hat, y_r, seed=seed)

    train_loader_t = DataLoader(train_set_t, batch_size=batch_size, shuffle=True)
    val_loader_t = DataLoader(val_set_t, batch_size=batch_size, shuffle=False)
    test_loader_t = DataLoader(test_set_t, batch_size=batch_size, shuffle=False)

    train_loader_s = DataLoader(train_set_s, batch_size=batch_size, shuffle=True)
    val_loader_s = DataLoader(val_set_s, batch_size=batch_size, shuffle=False)
    test_loader_s = DataLoader(test_set_s, batch_size=batch_size, shuffle=False)
    
    train_loader_r = DataLoader(train_set_r, batch_size=batch_size, shuffle=True)
    val_loader_r = DataLoader(val_set_r, batch_size=batch_size, shuffle=False)
    test_loader_r = DataLoader(test_set_r, batch_size=batch_size, shuffle=False)

    best_overall_val_loss = float('inf')
    best_model_settings = None
    best_model_paras = None

    for dim, depth, lr, lambda_orth, lambda_red in itertools.product(dims, depths, lrs, lambda_orths, lambda_reds):
        model = Integmodel(input_dim, input_dim_t, input_dim_s, input_dim_r, shared_out_dim=dim, unique_out_dim=dim, depth=depth)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)
        
        best_val_loss = float('inf')
        patience = 30  
        num_bad_epochs = 0  
        best_model = None  
        
        for epoch in range(num_epochs):
            model.train()  
            for ((X_t, y_t), (X_s, y_s), (X_r, y_r)) in zip(train_loader_t, train_loader_s, train_loader_r):
                optimizer.zero_grad()
                y_t_pred, y_s_pred, y_r_pred, orth, red = model(X_t, X_s, X_r)
                loss_t = criterion(y_t_pred, y_t)
                loss_s = criterion(y_s_pred, y_s)
                loss_r = criterion(y_r_pred, y_r)
                total_loss = loss_t + loss_s + loss_r + lambda_orth * orth + lambda_red * red 
                total_loss.backward()
                optimizer.step()
            scheduler.step()  

            model.eval()
            total_val_loss = 0
            val_batch = 0
            for ((X_t, y_t), (X_s, y_s), (X_r, y_r)) in zip(val_loader_t, val_loader_s, val_loader_r):
                with torch.no_grad():
                    y_t_pred, y_s_pred, y_r_pred, _, _ = model(X_t, X_s, X_r)
                    loss_t = criterion(y_t_pred, y_t)
                    loss_s = criterion(y_s_pred, y_s)
                    loss_r = criterion(y_r_pred, y_r)
                    total_val_loss += loss_t.item() + loss_s.item() +loss_r.item()
                    val_batch += 1
            average_val_loss = total_val_loss / val_batch
            
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                best_model = deepcopy(model) 
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
                if num_bad_epochs >= patience:
                    break 

        if best_val_loss < best_overall_val_loss:
            best_overall_val_loss = best_val_loss
            best_paras = [dim, depth, lr, lambda_orth, lambda_red]
            best_model_settings = best_model

    #print(best_paras)
    model=best_model_settings
    model.eval()
    test_loss_t = 0
    test_loss_s = 0
    test_loss_r = 0
    test_batch = 0
    
    with torch.no_grad():
        for ((X_t, y_t), (X_s, y_s), (X_r, y_r)) in zip(test_loader_t, test_loader_s, test_loader_r):
            y_t_pred, y_s_pred, y_r_pred, _, _ = model(X_t, X_s, X_r)
            test_loss_t += torch.sqrt(criterion(y_t_pred, y_t)).item()
            test_loss_s += torch.sqrt(criterion(y_s_pred, y_s)).item()
            test_loss_r += torch.sqrt(criterion(y_r_pred, y_r)).item()
            test_batch += 1
    
    average_test_loss_s = test_loss_s / test_batch  
    average_test_loss_t = test_loss_t / test_batch
    average_test_loss_r = test_loss_r / test_batch
    
    return average_test_loss_s, average_test_loss_t, average_test_loss_r  


results = []
num_seeds = 20
for seed in range(num_seeds): 
    set_seed(seed)
    
    p1, p2, p3, p4 = 125, 25, 25, 25
    p = p1 + p2 + p3 +p4
    n_t = 300
    n_s = 300
    n_r = 300
    
    covariance_matrix = torch.zeros(p,p)
    for i in range(p):
        for j in range(p):
            covariance_matrix[i, j] = 0.95**(0.01*abs(i-j))
    covariance_matrix.fill_diagonal_(1)
    
    dist = torch.distributions.MultivariateNormal(torch.zeros(p), covariance_matrix=covariance_matrix)
    X_t = dist.sample((n_t,))
    X_t_1 = X_t[:, :p1]
    X_t_2 = X_t[:, p1:p1+p2]
    X_t_3 = X_t[:, p1+p2:p1+p2+p3]
    X_t_4 = X_t[:, p1+p2+p3:]
    

    covariance_matrix = torch.zeros(p,p)
    for i in range(p):
        for j in range(p):
            covariance_matrix[i, j] = 0.9**(0.01*abs(i-j))
    covariance_matrix.fill_diagonal_(1)
    
    dist = torch.distributions.MultivariateNormal(torch.zeros(p), covariance_matrix=covariance_matrix)   
    X_s = dist.sample((n_s,))
    X_s_1 = X_s[:, :p1]
    X_s_2 = X_s[:, p1:p1+p2]
    X_s_3 = X_s[:, p1+p2:p1+p2+p3]
    X_s_4 = X_s[:, p1+p2+p3:]
    
    covariance_matrix = torch.zeros(p,p)
    for i in range(p):
        for j in range(p):
            covariance_matrix[i, j] = 0.9**(0.01*abs(i-j))
    covariance_matrix.fill_diagonal_(1)
    
    dist = torch.distributions.MultivariateNormal(torch.zeros(p), covariance_matrix=covariance_matrix)  
    X_r = dist.sample((n_r,))
    X_r_1 = X_r[:, :p1]
    X_r_2 = X_r[:, p1:p1+p2]
    X_r_3 = X_r[:, p1+p2:p1+p2+p3]
    X_r_4 = X_r[:, p1+p2+p3:]
    
    mean = -10
    std = 10
    v_c = torch.randn(p1 + p2 + p3 + p4) * std + mean
    v_t = torch.randn(p1 + p2 + p3 + p4) * std + mean
    v_s = torch.randn(p1 + p2 + p3 + p4) * std + mean
    v_r = torch.randn(p1 + p2 + p3 + p4) * std + mean
    epsilon_t = torch.randn(n_t)*0.1
    epsilon_s = torch.randn(n_s)*0.1
    epsilon_r = torch.randn(n_r)*0.1
    alpha=0.3
    y_t = alpha*(torch.sum((X_t**2) * v_c, dim=1) / (p1 + p2 + p3 + p4)) + (1-alpha)*(torch.sum((X_t) * v_t, dim=1) / (p1 + p2 + p3 + p4)) + epsilon_t
    y_s = alpha*(torch.sum((X_s**2) * v_c, dim=1) / (p1 + p2 + p3 + p4)) + (1-alpha)*(torch.sum((X_s) * v_s, dim=1) / (p1 + p2 + p3 + p4)) + epsilon_s
    y_r = alpha*(torch.sum((X_r**2) * v_c, dim=1) / (p1 + p2 + p3 + p4)) + (1-alpha)*(torch.sum((X_r) * v_r, dim=1) / (p1 + p2 + p3 + p4)) + epsilon_r
    y_t=y_t.view(-1,1)
    y_s=y_s.view(-1,1)
    y_r=y_r.view(-1,1)

    start_time = time.time()
    
    X_t_hat=torch.cat([X_t_1, X_t_2],dim=1)
    X_s_hat=torch.cat([X_s_1, X_s_3],dim=1)
    X_r_hat=torch.cat([X_r_1, X_r_4],dim=1)
    
    average_test_loss_s, average_test_loss_t, average_test_loss_r = grid_search_integmodel(X_t_hat, y_t, X_s_hat, y_s, X_r_hat, y_r, seed=seed)
    elapsed_time = time.time() - start_time
    results.append((seed, average_test_loss_s, average_test_loss_t, average_test_loss_r, elapsed_time))
       

df = pd.DataFrame(results, columns=['seed', 'average_test_loss_s', 'average_test_loss_t', 'average_test_loss_r', 'time'])
filename = f"HTL_three_data_gaussian_seeds={num_seeds}.csv"
filepath = os.path.join(".", filename)  
df.to_csv(filepath, index=False)
print(f"HTL Results saved to {filepath}")






