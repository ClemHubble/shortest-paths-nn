import argparse
from tqdm import tqdm, trange
from torch.optim import Adam, AdamW
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader

from src.baselines import *
from src.loss_funcs import *
import numpy as np
import torch
import torch.nn as nn
from torchmetrics.regression import MeanAbsolutePercentageError

from torch_geometric.utils import k_hop_subgraph
from torch.utils.data import Dataset, DataLoader
from torch_geometric.transforms import ToSparseTensor, VirtualNode, ToUndirected
from torch_geometric.nn import to_hetero
from src.transforms import add_laplace_positional_encoding, add_virtual_node
import yaml
import os
import csv
import logging
import copy

import time

from torch.profiler import profile, record_function, ProfilerActivity

import wandb

wandb.login()

MSE = nn.MSELoss()

output_dir = '/data/sam/terrain/'

sparse_tensor = ToSparseTensor()
virtual_node_transform = VirtualNode()

class SingleGraphShortestPathDataset(Dataset):
    def __init__(self, sources, targets, lengths):
        self.sources = sources
        self.targets = targets
        self.lengths = lengths

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx], self.targets[idx], self.lengths[idx]

class TerrainPatchesDataset(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'src':
            return self.x.size(0)
        if key == 'tar':
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)

def npz_to_dataset(data):
    
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long)

    srcs = torch.tensor(data['srcs'])
    tars = torch.tensor(data['tars'])
    lengths = torch.tensor(data['lengths'])
    node_features = torch.tensor(data['node_features'], dtype=torch.double)
    l2 = torch.norm(node_features[srcs] - node_features[tars], dim=1, p=2)

    train_dataset = SingleGraphShortestPathDataset(srcs, tars, lengths)

    return train_dataset, node_features, edge_index

def debug_dataset(data, n=100):
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long)

    srcs = torch.tensor(data['srcs'][:n])
    tars = torch.tensor(data['tars'][:n])
    lengths = torch.tensor(data['lengths'][:n])
    node_features = torch.tensor(data['node_features'], dtype=torch.double)
    l2 = torch.norm(node_features[srcs] - node_features[tars], dim=1, p=2)

    train_dataset = SingleGraphShortestPathDataset(srcs, tars, lengths)

    return train_dataset, node_features, edge_index

def format_log_dir(output_dir, 
                   dataset_name, 
                   siamese, 
                   modelname, 
                   vn, 
                   aggr, 
                   loss_func, 
                   layer_type,
                   p,
                   trial):
    log_dir = os.path.join(output_dir, 
                           'models',
                           'single_dataset', 
                           dataset_name, 
                           layer_type,
                           'vn' if vn else 'no-vn',
                           'siamese' if siamese else 'mlp',
                           f'p-{p}')
    if not siamese:
        log_dir = os.path.join(log_dir, aggr)
    log_dir = os.path.join(log_dir, loss_func, modelname, trial)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def configure_embedding_module(model_config, 
                               layer_type,
                               edge_dim=1,
                               new=True):
    print(model_config)
    print(new)
    embedding_config = model_config['constr']
    layer_norm = model_config['layer_norm']
    dropout = model_config['dropout']
    activation = model_config['activation']
    if not new and 'MLP' in layer_type:
        embedding_module = initialize_mlp(**embedding_config, 
                                          activation=activation, 
                                          layer_norm=layer_norm, 
                                          dropout=dropout)
    elif 'MLP' in layer_type and new:
        embedding_module = NewMLP(**embedding_config, add_norm=layer_norm)
    else:
        embedding_module = GNNModel(layer_type=layer_type, 
                                    edge_dim=edge_dim, 
                                    activation=activation, 
                                    layer_norm=layer_norm, 
                                    **embedding_config)
    return embedding_module

    
def configure_mlp_module(mlp_config, 
                        aggr='sum', 
                        new=True):
    model_config = copy.deepcopy(mlp_config['constr'])
    layer_norm=mlp_config['layer_norm']
    dropout=mlp_config['dropout']
    
    if aggr == 'combine':
        model_config['input'] = model_config['input'] * 3
    elif aggr == 'concat' or aggr == 'sum+diff':
        model_config['input'] = model_config['input'] * 2 
    if not new:
        mlp_nn = initialize_mlp(**model_config, layer_norm=False, dropout=dropout, activation='lrelu')
    else:
        mlp_nn = NewMLP(**model_config, add_norm=layer_norm)
    mlp = MLPBaseline1(mlp_nn, aggr=aggr)
    return mlp

def train_terrains_decoupled(train_dictionary,
                            model_config, 
                            layer_type, 
                            device,
                            prev_model_pth,
                            finetune_dataset_name,
                            activation='silu',
                            epochs=100, 
                            loss_func='mse_loss',
                            lr =0.001,
                            p=1, 
                            aggr='sum',
                            edge_attr=None, 
                            layer_norm=True,
                            new=True,
                            **kwargs):
    
    edge_dim=1
    embedding_config = model_config['gnn']
    embedding_module = configure_embedding_module(embedding_config, 
                                                 layer_type, 
                                                 edge_dim=edge_dim,
                                                 new=new)
    print(embedding_module)
    prev_model_state_pth = os.path.join(prev_model_pth, 'final_model.pt')
    print("Loading from:", prev_model_state_pth)
    embedding_model_state = torch.load(prev_model_state_pth, map_location='cpu')
    print(embedding_model_state.keys())
    embedding_module.load_state_dict(embedding_model_state)
    embedding_module.to(torch.double)
    
    print(model_config)
    print(embedding_module)

    
    mlp = configure_mlp_module(model_config['mlp'], aggr=aggr, new=new)
    mlp = mlp.to(torch.double)
    mlp.to(device)

    for param in embedding_module.parameters():
        param.requires_grad =False
    
    ## Pre-process all datasets
    num_graphs = len(train_dictionary['graphs'])
    train_dictionary['embeddings'] = []

    for i in range(num_graphs):
        graph_data = train_dictionary['graphs'][i]
        if 'MLP' in layer_type:
            node_embeddings = embedding_module(graph_data.x)
        else:
            node_embeddings = embedding_module(graph_data.x, graph_data.edge_index, edge_attr = graph_data.edge_attr)
        
        train_dictionary['embeddings'].append(node_embeddings)
        
    record_dir = os.path.join(prev_model_pth, finetune_dataset_name)
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    log_file = os.path.join(record_dir, 'training_log.log')
    logging.basicConfig(level=logging.INFO, filename=log_file)

    logging.info(f'GNN layer: {layer_type}')
    logging.info(f'Number of epochs: {epochs}')
    logging.info(f'MLP aggregation: {aggr}')
    logging.info(f'loss function: {loss_func}')

    run = wandb.init(
        project='terrains',
        dir='/data/sam/wandb',
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "p": p,
            "previous model path": prev_model_pth
        }
    )

    optimizer = AdamW(mlp.parameters(), lr=lr)

    for epoch in trange(epochs):
        total_loss = 0
        total_samples = 0
        for i in range(num_graphs):
            embeddings = train_dictionary['embeddings'][i]
            dataloader = train_dictionary['dataloaders'][i]
            total_samples = len(dataloader.dataset)
            for batch in dataloader:
                srcs = batch[0]
                tars = batch[1]
                embd_srcs = embeddings[srcs].to(device, non_blocking=True)
                embd_tars = embeddings[tars].to(device, non_blocking=True)
                lengths = batch[2].to(device, non_blocking=True)

                pred = mlp(embd_srcs, embd_tars, vn_emb=None)
                pred=pred.squeeze()

                loss = globals()[loss_func](pred, lengths)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.detach()
            normalized_abs_err = globals()['nmae_loss'](pred, lengths)
            wandb.log({'train_loss': loss, 
                    'total_loss': total_loss/total_samples,
                    'normalized_abs_err': normalized_abs_err})

    logging.info(f'final training loss:{total_loss/total_samples}')

    print("Final training loss:", total_loss/total_samples)
    path = os.path.join(record_dir, 'final_model.pt')
    print("saving model to:", path)
    torch.save({'gnn_state_dict':embedding_module.state_dict(), 
                'mlp_state_dict': mlp.state_dict()}, path)
    return embedding_module, mlp


# train_dictionary = {'graphs': [g1, g2, g3, ....], 'dataloaders': [dl1, dl2, ....]}
def train_few_cross_terrain_case(train_dictionary,
                                model_config, 
                                layer_type, 
                                device,
                                activation='silu',
                                epochs=100, 
                                loss_func='mse_loss',
                                lr =0.001,
                                log_dir='/data/sam',
                                siamese=True,
                                p=1, 
                                aggr='sum',
                                new=False,
                                finetune_from=None):
    torch.manual_seed(0)
    num_graphs = len(train_dictionary['graphs'])
    edge_dim = 1
    embedding_config = model_config['gnn']
    embedding_module = configure_embedding_module(embedding_config, 
                                                 layer_type, 
                                                 edge_dim=edge_dim,
                                                 new=new)
    print(embedding_module)
    if finetune_from:
        embedding_model_state = torch.load(finetune_from, map_location='cpu')
        embedding_module.load_state_dict(embedding_model_state)
        log_dir = os.path.join(log_dir, 'finetune/')
    embedding_module.to(torch.double)
    embedding_module.to(device)
    print(embedding_module)    
    mlp=None

    if siamese:
        parameters = embedding_module.parameters()
    else:
        mlp = configure_mlp_module(model_config['mlp'], aggr=aggr, new=new)
        mlp = mlp.to(torch.double)
        mlp.to(device)
        parameters = list(embedding_module.parameters()) + list(mlp.parameters())
        print(mlp)
    
    optimizer = AdamW(parameters, lr=lr)
    record_dir = os.path.join(log_dir, 'record/')
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    log_file = os.path.join(record_dir, 'training_log.log')
    logging.basicConfig(level=logging.INFO, filename=log_file)

    run = wandb.init(
        project='terrains',
        dir='/data/sam/wandb',
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "siamese": siamese,
            "p": p
        }
    )

    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    log_file = os.path.join(record_dir, 'training_log.log')
    logging.basicConfig(level=logging.INFO, filename=log_file)

    logging.info(f'GNN layer: {layer_type}')
    logging.info(f'Number of epochs: {epochs}')
    logging.info(f'MLP aggregation: {aggr}')
    logging.info(f'Siamese? {siamese}')
    logging.info(f'loss function: {loss_func}')
    print("logging to....", record_dir)
    for epoch in trange(epochs):
        total_loss = 0
        total_samples = 0
        for i in range(num_graphs):
            graph_data = train_dictionary['graphs'][i].to(device)
            dataloader = train_dictionary['dataloaders'][i]
            total_samples = len(dataloader.dataset)
            for batch in dataloader:
                srcs = batch[0].to(device, non_blocking=True)
                tars = batch[1].to(device, non_blocking=True)
                lengths = batch[2].to(device, non_blocking=True)
                if 'MLP' in layer_type:
                    src_embeddings = embedding_module(graph_data.x[srcs])
                    tar_embeddings = embedding_module(graph_data.x[tars])
                else:
                    node_embeddings = embedding_module(graph_data.x,graph_data.edge_index,edge_attr = graph_data.edge_attr)

                    src_embeddings = node_embeddings[srcs]
                    tar_embeddings = node_embeddings[tars]
                    
                if siamese:
                    pred = torch.norm(src_embeddings - tar_embeddings, p=p, dim=1)
                else:
                    pred = mlp(src_embeddings, tar_embeddings, vn_emb=None)
                    pred=pred.squeeze()
                loss = globals()[loss_func](pred, lengths)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.detach()
        
        normalized_abs_err = globals()['nmae_loss'](pred, lengths)
        wandb.log({'train_loss': loss, 
                   'total_loss': total_loss/total_samples,
                   'normalized_abs_err': normalized_abs_err})

    logging.info(f'final training loss:{total_loss/total_samples}')
    print("Final training loss:", total_loss/total_samples)
    print("siamese", siamese)
    if siamese:
        path = os.path.join(log_dir, 'final_model.pt')
        print("saving model to:", path)
        torch.save(embedding_module.state_dict(), path)
        return embedding_module
    else:
        
        path = os.path.join(log_dir, 'final_model.pt')
        print("saving model to:", path)
        torch.save({'gnn_state_dict':embedding_module.state_dict(), 
                    'mlp_state_dict': mlp.state_dict()}, path)
        return embedding_module, mlp
