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

class FlexibleMLP(nn.Module):
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


def grid_search_integmodel(X_t_hat, y_t, p, seed):
    dims = [32, 64, 128]
    depths = [2, 3, 4]
    lrs = [0.001]
    batch_size = 16 
    num_epochs = 25000
    input_dim=p
    
    train_set_t, val_set_t, test_set_t = prepare_data(X_t_hat, y_t, seed=seed) 

    train_loader_t = DataLoader(train_set_t, batch_size=batch_size, shuffle=True)
    val_loader_t = DataLoader(val_set_t, batch_size=batch_size, shuffle=False)
    test_loader_t = DataLoader(test_set_t, batch_size=batch_size, shuffle=False)

    best_overall_val_loss = float('inf')
    best_model_settings = None
    best_model_paras = None
 
    for dim, depth, lr in itertools.product(dims, depths, lrs):
        model = FlexibleMLP(input_dim,  depth, dim) 
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)
        
        best_val_loss = float('inf')
        patience = 30  
        num_bad_epochs = 0  
        best_model = None  
        
        for epoch in range(num_epochs):
            model.train()  
            for (X_t, y_t) in train_loader_t:
                optimizer.zero_grad()
                y_t_pred = model(X_t)
                loss_t = criterion(y_t_pred, y_t)
                loss_t.backward()
                optimizer.step()
            scheduler.step()   
            
            model.eval()
            total_val_loss = 0
            val_batch = 0
            for (X_t, y_t) in val_loader_t:
                with torch.no_grad():
                    y_t_pred = model(X_t)
                    loss_t = criterion(y_t_pred, y_t)
                    total_val_loss += loss_t.item()
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
            best_paras = [dim, depth, lr]
            best_model_settings = best_model

    #print(best_paras)
    model=best_model_settings
    model.eval()
    test_loss_t = 0
    test_loss_s = 0
    test_batch = 0
    
    with torch.no_grad():
        for (X_t, y_t) in test_loader_t:
            y_t_pred = model(X_t)
            test_loss_t += torch.sqrt(criterion(y_t_pred, y_t)).item()
            test_batch += 1
    
    average_test_loss_t = test_loss_t / test_batch  
    
    return average_test_loss_t  

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
    X_t = X_t + 0.3 * torch.sin(X_t)
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
    X_s = X_s + 0.3 * torch.sin(X_s)
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
    X_r = X_r + 0.3 * torch.sin(X_r)
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
    X_s_hat=torch.cat([X_s_1, X_s_3],dim=1)
    average_test_loss_s = grid_search_integmodel(X_s_hat, y_s, p1+p3, seed=seed)
    
    X_t_hat=torch.cat([X_t_1, X_t_2],dim=1)
    average_test_loss_t = grid_search_integmodel(X_t_hat, y_t, p1+p2, seed=seed)
    
    X_r_hat=torch.cat([X_r_1, X_r_4],dim=1)
    average_test_loss_r = grid_search_integmodel(X_r_hat, y_r, p1+p4, seed=seed)

    elapsed_time = time.time() - start_time
    results.append((seed, average_test_loss_s, average_test_loss_t, average_test_loss_r, elapsed_time))
        


df = pd.DataFrame(results, columns=['seed', 'average_test_loss_s', 'average_test_loss_t', 'average_test_loss_r', 'time'])
filename = f"SDL_three_data_nongaussian_seeds={num_seeds}.csv"
filepath = os.path.join(".", filename)  
df.to_csv(filepath, index=False)
print(f"SDL Results saved to {filepath}")


