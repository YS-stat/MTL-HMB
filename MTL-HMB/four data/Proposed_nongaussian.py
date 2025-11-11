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
    def __init__(self, input_dim, output_dim, depth, width):
        super(FlexibleMLP, self).__init__()
        layers = []
        for i in range(depth):
            if i < depth - 1:  # Add ReLU layer only if it's not the last layer
                layers.append(nn.Sequential(
                    nn.Linear(input_dim, width),
                    nn.ReLU()
                ))
                input_dim = width
            else:
                # The last layer does not get a ReLU regardless of the depth
                layers.append(nn.Linear(input_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def imputation_grid_search(X_t_1_train, X_t_1_val, X_s_1_train, X_s_1_val,                            X_t_2_train, X_t_2_val, X_s_1, input_dim, output_dim, seed):
    set_seed(seed)
    widths = [4,8,16] #[4,8,16,32]
    depths = [1,2] #[1,2,3]
    lrs = [0.001]
    lambda_orths = [0]
    
    best_eval_loss = float("inf")
    best_model_state = None
    best_config = None

    for width, depth, lr, lambda_orth in itertools.product(widths, depths, lrs, lambda_orths):
        shared_feature_dim = width
        unique_feature_dim = width
        
        shared_encoder = FlexibleMLP(input_dim,   shared_feature_dim, depth, width) 
        unique_encoder_t = FlexibleMLP(input_dim, unique_feature_dim, depth, width)
        unique_encoder_s = FlexibleMLP(input_dim, unique_feature_dim, depth, width)
        decoder = FlexibleMLP(shared_feature_dim + unique_feature_dim, input_dim, depth, input_dim)
        predictor = FlexibleMLP(shared_feature_dim, output_dim, depth, width) 
        
        loss_fn = torch.nn.MSELoss()
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
            
            total_loss = 1*(loss_X_t_1 + loss_X_s_1) + loss_X_t_2 + lambda_orth * orth
            total_loss.backward()
            optimizer.step()

            shared_encoder.eval()
            unique_encoder_t.eval()
            unique_encoder_s.eval()
            decoder.eval()
            predictor.eval()

            with torch.no_grad():
                f_t_c = shared_encoder(X_t_1_val) 
                X_t_2_hat = predictor(f_t_c)
                eval_loss = loss_fn(X_t_2_hat, X_t_2_val)
                
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

    #print(best_config)
    shared_encoder = best_model['shared_encoder']
    predictor = best_model['predictor']

    shared_encoder.eval()  
    predictor.eval()   
    with torch.no_grad():
        f_s_c = shared_encoder(X_s_1) 
        X_s_2_hat_best = predictor(f_s_c)  

    return X_s_2_hat_best 


class Integmodel(nn.Module):
    def __init__(self, input_dim, input_dim_t, input_dim_s, input_dim_r, input_dim_u, shared_out_dim, unique_out_dim, depth):
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
        
        self.unique_encoder_u = create_layers(input_dim_u, unique_out_dim, depth)

        self.g = create_layers_y(shared_out_dim + unique_out_dim, shared_out_dim, depth)
        
        self.g_t = create_layers_y_unique(shared_out_dim + unique_out_dim, unique_out_dim, shared_out_dim, depth)

        self.g_s = create_layers_y_unique(shared_out_dim + unique_out_dim, unique_out_dim, shared_out_dim, depth)
        
        self.g_r = create_layers_y_unique(shared_out_dim + unique_out_dim, unique_out_dim, shared_out_dim, depth)
        
        self.g_u = create_layers_y_unique(shared_out_dim + unique_out_dim, unique_out_dim, shared_out_dim, depth)
    
    def forward(self, x_t, x_s, x_r, x_u):
        f_t_c = self.shared_encoder(x_t[:, :p1])
        f_s_c = self.shared_encoder(x_s[:, :p1])
        f_r_c = self.shared_encoder(x_r[:, :p1])
        f_u_c = self.shared_encoder(x_u[:, :p1])

        f_t_u = self.unique_encoder_t(x_t[:, p1:])
        f_s_u = self.unique_encoder_s(x_s[:, p1:])
        f_r_u = self.unique_encoder_r(x_r[:, p1:])
        f_u_u = self.unique_encoder_u(x_u[:, p1:])

        g_input_s = torch.cat([f_s_u, f_s_c], dim=1)
        g_input_t = torch.cat([f_t_u, f_t_c], dim=1)
        g_input_r = torch.cat([f_r_u, f_r_c], dim=1)
        g_input_u = torch.cat([f_u_u, f_u_c], dim=1)
 
        h_t = self.g_t[0](g_input_t)
        h_s = self.g_s[0](g_input_s)
        h_r = self.g_r[0](g_input_r)
        h_u = self.g_u[0](g_input_u) 
        h_t_o = self.g[0](g_input_t)
        h_s_o = self.g[0](g_input_s)
        h_r_o = self.g[0](g_input_r)
        h_u_o = self.g[0](g_input_u)

        norm_products = 0
        for i in range(1, self.depth):
            h_t_next_input = torch.cat([h_t, h_t_o], dim=1)  
            h_s_next_input = torch.cat([h_s, h_s_o], dim=1)
            h_r_next_input = torch.cat([h_r, h_r_o], dim=1)
            h_u_next_input = torch.cat([h_u, h_u_o], dim=1)
            
            h_t = self.g_t[i](h_t_next_input) 
            h_s = self.g_s[i](h_s_next_input)
            h_r = self.g_r[i](h_r_next_input)
            h_u = self.g_u[i](h_u_next_input)
            h_t_o = self.g[i](h_t_o) 
            h_s_o = self.g[i](h_s_o)
            h_r_o = self.g[i](h_r_o)
            h_u_o = self.g[i](h_u_o)
            
            half_input_size = h_t_o.size(1)
            weight_product = torch.matmul(self.g_s[i][0].weight[:, -half_input_size:], self.g[i][0].weight.t()) +                              torch.matmul(self.g_t[i][0].weight[:, -half_input_size:], self.g[i][0].weight.t()) +                              torch.matmul(self.g_r[i][0].weight[:, -half_input_size:], self.g[i][0].weight.t()) +                              torch.matmul(self.g_u[i][0].weight[:, -half_input_size:], self.g[i][0].weight.t())

            norm_products += (torch.norm(weight_product))**2

        y_s= self.g_s[-1](torch.cat([h_s, h_s_o], dim=1))+self.g[-1](h_s_o)
        y_t= self.g_t[-1](torch.cat([h_t, h_t_o], dim=1))+self.g[-1](h_t_o)
        y_r= self.g_r[-1](torch.cat([h_r, h_r_o], dim=1))+self.g[-1](h_r_o)
        y_u= self.g_u[-1](torch.cat([h_u, h_u_o], dim=1))+self.g[-1](h_u_o)
        
        half_input_size = h_t_o.size(1)
        weight_product = torch.matmul(self.g_s[-1].weight[:, -half_input_size:], self.g[-1].weight.t()) +                          torch.matmul(self.g_t[-1].weight[:, -half_input_size:], self.g[-1].weight.t()) +                          torch.matmul(self.g_r[-1].weight[:, -half_input_size:], self.g[-1].weight.t()) +                          torch.matmul(self.g_u[-1].weight[:, -half_input_size:], self.g[-1].weight.t())
        norm_products += (torch.norm(weight_product))**2
        

        orth = torch.norm(f_t_c.t() @ f_t_u)**2 + torch.norm(f_s_c.t() @ f_s_u)**2 + torch.norm(f_r_c.t() @ f_r_u)**2 + torch.norm(f_u_c.t() @ f_u_u)**2
        rob_penalty = 1*torch.sum(self.unique_encoder_t[0][0].weight[:, p1+p2:]**2) +                       1*torch.sum(self.unique_encoder_s[0][0].weight[:, p1:p1+p2]**2) +                       1*torch.sum(self.unique_encoder_s[0][0].weight[:, p1+p2+p3:]**2) +                       1*torch.sum(self.unique_encoder_r[0][0].weight[:, p1:p1+p2+p3]**2) +                       1*torch.sum(self.unique_encoder_r[0][0].weight[:, p1+p2+p3+p4:]**2) +                       1*torch.sum(self.unique_encoder_u[0][0].weight[:, p1:p1+p2+p3+p4]**2)
    

        return y_t, y_s, y_r, y_u, orth, rob_penalty, norm_products 


def grid_search_integmodel(X_t_hat, y_t, X_s_hat, y_s, X_r_hat, y_r, X_u_hat, y_u, seed):
    set_seed(seed)
    dims = [32, 64, 128]
    depths = [2,3,4]
    lrs = [0.001]
    lambda_orths =[0.01,0.1] 
    lambda_robs = [0.01,0.1]
    lambda_reds = [0.01,0.1]
    batch_size = 16 
    num_epochs = 25000
    input_dim=p1
    input_dim_t=p1+p2+p3+p4+p5
    input_dim_s=p1+p2+p3+p4+p5
    input_dim_r=p1+p2+p3+p4+p5
    input_dim_u=p1+p2+p3+p4+p5
    
    train_set_t, val_set_t, test_set_t = prepare_data(X_t_hat, y_t, seed=seed)
    train_set_s, val_set_s, test_set_s = prepare_data(X_s_hat, y_s, seed=seed)
    train_set_r, val_set_r, test_set_r = prepare_data(X_r_hat, y_r, seed=seed)
    train_set_u, val_set_u, test_set_u = prepare_data(X_u_hat, y_u, seed=seed)

    train_loader_t = DataLoader(train_set_t, batch_size=batch_size, shuffle=True)
    val_loader_t = DataLoader(val_set_t, batch_size=batch_size, shuffle=False)
    test_loader_t = DataLoader(test_set_t, batch_size=batch_size, shuffle=False)

    train_loader_s = DataLoader(train_set_s, batch_size=batch_size, shuffle=True)
    val_loader_s = DataLoader(val_set_s, batch_size=batch_size, shuffle=False)
    test_loader_s = DataLoader(test_set_s, batch_size=batch_size, shuffle=False)
    
    train_loader_r = DataLoader(train_set_r, batch_size=batch_size, shuffle=True)
    val_loader_r = DataLoader(val_set_r, batch_size=batch_size, shuffle=False)
    test_loader_r = DataLoader(test_set_r, batch_size=batch_size, shuffle=False)
    
    train_loader_u = DataLoader(train_set_u, batch_size=batch_size, shuffle=True)
    val_loader_u = DataLoader(val_set_u, batch_size=batch_size, shuffle=False)
    test_loader_u = DataLoader(test_set_u, batch_size=batch_size, shuffle=False)

    best_overall_val_loss = float('inf')
    best_model_settings = None
    best_model_paras = None

    for dim, depth, lr, lambda_orth, lambda_rob, lambda_red in itertools.product(dims, depths, lrs, lambda_orths, lambda_robs, lambda_reds):
        model = Integmodel(input_dim, input_dim_t, input_dim_s, input_dim_r, input_dim_u, shared_out_dim=dim, unique_out_dim=dim, depth=depth)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)
        
        best_val_loss = float('inf')
        patience = 30  
        num_bad_epochs = 0  
        best_model = None 
        
        for epoch in range(num_epochs):
            model.train()  
            for ((X_t, y_t), (X_s, y_s), (X_r, y_r), (X_u, y_u)) in zip(train_loader_t, train_loader_s, train_loader_r, train_loader_u):
                optimizer.zero_grad()
                y_t_pred, y_s_pred, y_r_pred, y_u_pred, orth, rob, red = model(X_t, X_s, X_r, X_u)
                loss_t = criterion(y_t_pred, y_t)
                loss_s = criterion(y_s_pred, y_s)
                loss_r = criterion(y_r_pred, y_r)
                loss_u = criterion(y_u_pred, y_u)
                total_loss = loss_t + loss_s + loss_r + loss_u + lambda_orth * orth + lambda_rob * rob + lambda_red * red 
                total_loss.backward()
                optimizer.step()
            scheduler.step()  

            model.eval()
            total_val_loss = 0
            val_batch = 0
            for ((X_t, y_t), (X_s, y_s), (X_r, y_r), (X_u, y_u)) in zip(val_loader_t, val_loader_s, val_loader_r, val_loader_u):
                with torch.no_grad():
                    y_t_pred, y_s_pred, y_r_pred, y_u_pred, _, _ , _= model(X_t, X_s, X_r, X_u)
                    loss_t = criterion(y_t_pred, y_t)
                    loss_s = criterion(y_s_pred, y_s)
                    loss_r = criterion(y_r_pred, y_r)
                    loss_u = criterion(y_u_pred, y_u)
                    total_val_loss += loss_t.item() + loss_s.item() + loss_r.item() + loss_u.item()
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
            best_paras = [dim, depth, lr, lambda_orth, lambda_rob, lambda_red]
            best_model_settings = best_model

    #print(best_paras)
    model=best_model_settings
    model.eval()
    test_loss_t = 0
    test_loss_s = 0
    test_loss_r = 0
    test_loss_u = 0
    test_batch = 0
    
    with torch.no_grad():
        for ((X_t, y_t), (X_s, y_s), (X_r, y_r), (X_u, y_u)) in zip(test_loader_t, test_loader_s, test_loader_r, test_loader_u):
            y_t_pred, y_s_pred, y_r_pred, y_u_pred, _, _, _ = model(X_t, X_s, X_r, X_u)
            test_loss_t += torch.sqrt(criterion(y_t_pred, y_t)).item()
            test_loss_s += torch.sqrt(criterion(y_s_pred, y_s)).item()
            test_loss_r += torch.sqrt(criterion(y_r_pred, y_r)).item()
            test_loss_u += torch.sqrt(criterion(y_u_pred, y_u)).item()
            test_batch += 1
    
    average_test_loss_s = test_loss_s / test_batch  
    average_test_loss_t = test_loss_t / test_batch
    average_test_loss_r = test_loss_r / test_batch
    average_test_loss_u = test_loss_u / test_batch
    
    return average_test_loss_s, average_test_loss_t, average_test_loss_r, average_test_loss_u  


results = []
num_seeds = 20
for seed in range(num_seeds):  
    set_seed(seed)
    
    p1, p2, p3, p4, p5 = 125, 25, 25, 25, 25
    p = p1 + p2 + p3 +p4 + p5
    n_t = 300
    n_s = 300
    n_r = 300
    n_u = 300
    
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
    X_t_4 = X_t[:, p1+p2+p3:p1+p2+p3+p4]
    X_t_5 = X_t[:, p1+p2+p3+p4:]
    

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
    X_s_4 = X_s[:, p1+p2+p3:p1+p2+p3+p4]
    X_s_5 = X_s[:, p1+p2+p3+p4:]
    
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
    X_r_4 = X_r[:, p1+p2+p3:p1+p2+p3+p4]
    X_r_5 = X_r[:, p1+p2+p3+p4:]
    
    covariance_matrix = torch.zeros(p,p)
    for i in range(p):
        for j in range(p):
            covariance_matrix[i, j] = 0.9**(0.01*abs(i-j))
    covariance_matrix.fill_diagonal_(1)
    
    dist = torch.distributions.MultivariateNormal(torch.zeros(p), covariance_matrix=covariance_matrix)  
    X_u = dist.sample((n_u,))
    X_u = X_u + 0.3 * torch.sin(X_u)
    X_u_1 = X_u[:, :p1]
    X_u_2 = X_u[:, p1:p1+p2]
    X_u_3 = X_u[:, p1+p2:p1+p2+p3]
    X_u_4 = X_u[:, p1+p2+p3:p1+p2+p3+p4]
    X_u_5 = X_u[:, p1+p2+p3+p4:]
    
    mean = -10
    std = 10
    v_c = torch.randn(p1 + p2 + p3 + p4 + p5) * std + mean
    v_t = torch.randn(p1 + p2 + p3 + p4 + p5) * std + mean
    v_s = torch.randn(p1 + p2 + p3 + p4 + p5) * std + mean
    v_r = torch.randn(p1 + p2 + p3 + p4 + p5) * std + mean
    v_u = torch.randn(p1 + p2 + p3 + p4 + p5) * std + mean
    epsilon_t = torch.randn(n_t)*0.1
    epsilon_s = torch.randn(n_s)*0.1
    epsilon_r = torch.randn(n_r)*0.1
    epsilon_u = torch.randn(n_u)*0.1
    alpha=0.3
    y_t = alpha*(torch.sum((X_t**2) * v_c, dim=1) / (p1 + p2 + p3 + p4 + p5)) + (1-alpha)*(torch.sum((X_t) * v_t, dim=1) / (p1 + p2 + p3 + p4 + p5)) + epsilon_t
    y_s = alpha*(torch.sum((X_s**2) * v_c, dim=1) / (p1 + p2 + p3 + p4 + p5)) + (1-alpha)*(torch.sum((X_s) * v_s, dim=1) / (p1 + p2 + p3 + p4 + p5)) + epsilon_s
    y_r = alpha*(torch.sum((X_r**2) * v_c, dim=1) / (p1 + p2 + p3 + p4 + p5)) + (1-alpha)*(torch.sum((X_r) * v_r, dim=1) / (p1 + p2 + p3 + p4 + p5)) + epsilon_r
    y_u = alpha*(torch.sum((X_u**2) * v_c, dim=1) / (p1 + p2 + p3 + p4 + p5)) + (1-alpha)*(torch.sum((X_u) * v_u, dim=1) / (p1 + p2 + p3 + p4 + p5)) + epsilon_u
    y_t=y_t.view(-1,1)
    y_s=y_s.view(-1,1)
    y_r=y_r.view(-1,1)
    y_u=y_u.view(-1,1)
    start_time = time.time()
    
    #imputation X_s_2  
    X_t_1_train, X_t_1_val = train_test_split(X_t_1, test_size=0.1, random_state=42)
    X_t_2_train, X_t_2_val = train_test_split(X_t_2, test_size=0.1, random_state=42)
    X_sru_1 = torch.cat([X_s_1, X_r_1, X_u_1],dim=0)
    X_sru_1_train, X_sru_1_val = train_test_split(X_sru_1, test_size=0.1, random_state=42)
   
    X_sru_2_hat_best=imputation_grid_search(X_t_1_train, X_t_1_val, X_sru_1_train,                                           X_sru_1_val, X_t_2_train, X_t_2_val, X_sru_1, input_dim=p1, output_dim=p2, seed=seed)
    X_s_2_hat_best=X_sru_2_hat_best[:n_s,:]
    X_r_2_hat_best=X_sru_2_hat_best[n_s:n_s+n_r,:]
    X_u_2_hat_best=X_sru_2_hat_best[n_s+n_r:,:]
    
    #imputation X_t_3
    X_s_1_train, X_s_1_val = train_test_split(X_s_1, test_size=0.1, random_state=42)
    X_s_3_train, X_s_3_val = train_test_split(X_s_3, test_size=0.1, random_state=42)
    X_tru_1 = torch.cat([X_t_1, X_r_1, X_u_1],dim=0)
    X_tru_1_train, X_tru_1_val = train_test_split(X_tru_1, test_size=0.1, random_state=42)
   
    X_tru_3_hat_best=imputation_grid_search(X_s_1_train, X_s_1_val, X_tru_1_train,                                           X_tru_1_val, X_s_3_train, X_s_3_val, X_tru_1, input_dim=p1, output_dim=p3, seed=seed)
    X_t_3_hat_best=X_tru_3_hat_best[:n_t,:]
    X_r_3_hat_best=X_tru_3_hat_best[n_t:n_t+n_r,:]
    X_u_3_hat_best=X_tru_3_hat_best[n_t+n_r:,:]

    # imputation
    X_r_1_train, X_r_1_val = train_test_split(X_r_1, test_size=0.1, random_state=42)
    X_r_4_train, X_r_4_val = train_test_split(X_r_4, test_size=0.1, random_state=42)
    X_tsu_1 = torch.cat([X_t_1, X_s_1, X_u_1],dim=0)
    X_tsu_1_train, X_tsu_1_val = train_test_split(X_tsu_1, test_size=0.1, random_state=42)
   
    X_tsu_4_hat_best=imputation_grid_search(X_r_1_train, X_r_1_val, X_tsu_1_train,                                           X_tsu_1_val, X_r_4_train, X_r_4_val, X_tsu_1, input_dim=p1, output_dim=p4, seed=seed)
    X_t_4_hat_best=X_tsu_4_hat_best[:n_t,:]
    X_s_4_hat_best=X_tsu_4_hat_best[n_t:n_t+n_s,:]   
    X_u_4_hat_best=X_tsu_4_hat_best[n_t+n_s:,:]  
    
    # imputation
    X_u_1_train, X_u_1_val = train_test_split(X_u_1, test_size=0.1, random_state=42)
    X_u_5_train, X_u_5_val = train_test_split(X_u_5, test_size=0.1, random_state=42)
    X_tsr_1 = torch.cat([X_t_1, X_s_1, X_r_1],dim=0)
    X_tsr_1_train, X_tsr_1_val = train_test_split(X_tsr_1, test_size=0.1, random_state=42)
   
    X_tsr_5_hat_best=imputation_grid_search(X_u_1_train, X_u_1_val, X_tsr_1_train,                                           X_tsr_1_val, X_u_5_train, X_u_5_val, X_tsr_1, input_dim=p1, output_dim=p5, seed=seed)
    X_t_5_hat_best=X_tsr_5_hat_best[:n_t,:]
    X_s_5_hat_best=X_tsr_5_hat_best[n_t:n_t+n_s,:]   
    X_r_5_hat_best=X_tsr_5_hat_best[n_t+n_s:,:]  
    
    X_t_hat=torch.cat([X_t_1, X_t_1, X_t_2, X_t_3_hat_best, X_t_4_hat_best, X_t_5_hat_best],dim=1)
    X_s_hat=torch.cat([X_s_1, X_s_1, X_s_2_hat_best, X_s_3, X_s_4_hat_best, X_s_5_hat_best],dim=1)
    X_r_hat=torch.cat([X_r_1, X_r_1, X_r_2_hat_best, X_r_3_hat_best, X_r_4, X_r_5_hat_best],dim=1)
    X_u_hat=torch.cat([X_u_1, X_u_1, X_u_2_hat_best, X_u_3_hat_best, X_u_4_hat_best, X_u_5],dim=1)
    
    average_test_loss_s, average_test_loss_t, average_test_loss_r, average_test_loss_u = grid_search_integmodel(X_t_hat, y_t, X_s_hat, y_s, X_r_hat, y_r, X_u_hat, y_u, seed=seed)
    elapsed_time = time.time() - start_time
    results.append((seed, average_test_loss_s, average_test_loss_t, average_test_loss_r, average_test_loss_u, elapsed_time))

        
df = pd.DataFrame(results, columns=['seed', 'average_test_loss_s', 'average_test_loss_t', 'average_test_loss_r', 'average_test_loss_u', 'time'])
filename = f"new_proposed_four_data_nongaussian_seeds={num_seeds}.csv"
filepath = os.path.join(".", filename)  
df.to_csv(filepath, index=False)
print(f"Proposed Results saved to {filepath}")




