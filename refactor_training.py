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

import time

from torch.profiler import profile, record_function, ProfilerActivity

import wandb

wandb.login()

MSE = nn.MSELoss()

output_dir = '/data/sam/terrain/'

sparse_tensor = ToSparseTensor()
virtual_node_transform = VirtualNode()

class SingleGraphShortestPathDataset(Dataset):
    def __init__(self, sources, targets, lengths, l2):
        self.sources = sources
        self.targets = targets
        self.lengths = lengths
        self.l2 = l2

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx], self.targets[idx], self.lengths[idx], self.l2[idx]

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

    train_dataset = SingleGraphShortestPathDataset(srcs, tars, lengths, l2)

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
                               activation='lrelu', 
                               edge_dim=1, 
                               layer_norm=True):
    print(model_config)
    embedding_config = model_config['gnn']
    if layer_type == 'MLP' or layer_type=='SiLUMLP':
        embedding_module = initialize_mlp(**embedding_config, activation=activation)
    elif layer_type == 'NewMLP':
        embedding_module = NewMLP(**embedding_config, add_norm=layer_norm)
    else:
        embedding_module = GNNModel(layer_type=layer_type, 
                                    edge_dim=edge_dim, 
                                    activation='SiLU', 
                                    layer_norm=layer_norm, 
                                    **embedding_config)
    return embedding_module

    
def configure_mlp_module(mlp_config, aggr='sum'):
    print(mlp_config)
    if aggr == 'combine':
        mlp_config['input'] = mlp_config['input'] * 3
    elif aggr == 'concat' or aggr == 'sum+diff':
        mlp_config['input'] = mlp_config['input'] * 2 
    
    mlp_nn = NewMLP(**mlp_config)
    mlp = MLPBaseline1(mlp_nn, aggr=aggr)
    return mlp

def train_single_terrain_e2e(node_features, 
                            edge_index, 
                            train_dataloader,
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
                            edge_attr=None, 
                            layer_norm=True):
    
    
    
    edge_attr = torch.tensor(edge_attr) if edge_attr is not None else None
    edge_attr = edge_attr.unsqueeze(-1)
    edge_dim = 1 if edge_attr is not None else 0
    graph_data = Data(x = node_features, edge_index=edge_index, edge_attr=edge_attr)

    embedding_config = model_config['gnn']
    embedding_module = configure_embedding_module(model_config, 
                                                 layer_type, 
                                                 activation=activation, 
                                                 edge_dim=edge_dim, 
                                                 layer_norm=layer_norm)
    embedding_module.to(torch.double)
    embedding_module.to(device)
    
    print(model_config)
    print(embedding_module)
    
    # if isinstance(embedding_module, GNNModel):
    graph_data =graph_data.to(device)
    record_dir = os.path.join(log_dir, 'record/')
    log_file = os.path.join(record_dir, 'training_log.log')
    logging.basicConfig(level=logging.INFO, filename=log_file)

    run = wandb.init(
        project='terrains',
        dir='/data/sam/wandb',
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "siamese": siamese,
            "p": p, 
            "layer_norm": layer_norm
        }
    )

    mlp=None
    if siamese:
        parameters = embedding_module.parameters()
    else:
        mlp = configure_mlp_module(model_config['mlp'], aggr=aggr)
        mlp = mlp.to(torch.double)
        mlp.to(device)
        parameters = list(embedding_module.parameters()) + list(mlp.parameters())
    
    record_dir = os.path.join(log_dir, 'record/')
    optimizer = AdamW(parameters, lr=lr)

    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    log_file = os.path.join(record_dir, 'training_log.log')
    logging.basicConfig(level=logging.INFO, filename=log_file)

    logging.info(f'GNN layer: {layer_type}')
    logging.info(f'Number of epochs: {epochs}')
    logging.info(f'MLP aggregation: {aggr}')
    logging.info(f'Siamese? {siamese}')
    logging.info(f'loss function: {loss_func}')
    logging.info(f'edge attributes?: {edge_attr != None}')

    total_samples = len(train_dataloader.dataset)
    for epoch in trange(epochs):
        total_loss = 0
        for batch in train_dataloader:
            srcs = batch[0].to(device, non_blocking=True)
            tars = batch[1].to(device, non_blocking=True)
            lengths = batch[2].to(device, non_blocking=True)

            if isinstance(embedding_module, NewMLP) or isinstance(embedding_module, MLP):
                src_embeddings = embedding_module(graph_data.x[srcs])
                tar_embeddings = embedding_module(graph_data.x[tars])
            else:
                node_embeddings = embedding_module(graph_data.x,graph_data.edge_index,edge_attr = graph_data.edge_attr)

                src_embeddings = node_embeddings[srcs]
                tar_embeddings = node_embeddings[tars]
            
            if siamese:
                pred = torch.norm(src_embeddings - tar_embeddings, p=p, dim=1)
            else:
                pred = mlp(src_embeddings - tar_embeddings, vn_emb=None)
            
            loss = globals()[loss_func](pred, lengths)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.detach()
        

        wandb.log({'train_loss': loss})

    logging.info(f'final training loss:{total_loss/total_samples}')
    print("Final training loss:", total_loss/total_samples)

    if siamese:
        path = os.path.join(log_dir, 'final_model.pt')
        print("saving model to:", path)
        torch.save(embedding_module.state_dict(), path)
        return embedding_module
    else:
        path = os.path.join(log_dir, 'final_model.pt')
        torch.save({'gnn_state_dict':embedding_module.state_dict(), 
                    'mlp_state_dict': mlp.state_dict()}, path)
        return embedding_module, mlp

def train_single_terrain_frozen(node_features, 
                                edge_index, 
                                train_dataloader,
                                model_config, 
                                layer_type, 
                                device,
                                prev_model_pth,
                                finetune_dataset_name,
                                activation='silu',
                                epochs=100, 
                                loss_func='mse_loss',
                                lr =0.001,
                                log_dir='/data/sam',
                                siamese=True,
                                p=1, 
                                aggr='sum',
                                edge_attr=None, 
                                layer_norm=True):
    
    edge_attr = torch.tensor(edge_attr) if edge_attr is not None else None
    edge_attr = edge_attr.unsqueeze(-1)
    edge_dim = 1 if edge_attr is not None else 0
    graph_data = Data(x = node_features, edge_index=edge_index, edge_attr=edge_attr)

    embedding_config = model_config['gnn']
    embedding_module = configure_embedding_module(model_config, 
                                                 layer_type, 
                                                 activation=activation, 
                                                 edge_dim=edge_dim, 
                                                 layer_norm=layer_norm)
    print("Loading from:" prev_model_pth)
    embedding_model_state = torch.load(prev_model_pth, map_location='cpu')
    embedding_module.to(torch.double)
    embedding_module.to(device)
    
    print(model_config)
    print(embedding_module)

    mlp = configure_mlp_module(model_config['mlp'], aggr=aggr)
    mlp = mlp.to(torch.double)
    mlp.to(device)





def train_few_cross_terrain_case(node_features, 
                                edge_index, 
                                train_dataloader,
                                model_config, 
                                layer_type, 
                                device,
                                activation='silu',
                                epochs=100, 
                                loss_fnc='mse_loss',
                                lr =0.001,
                                log_dir='/data/sam',
                                siamese=True,
                                p=1, 
                                aggr='sum',
                                edge_attr=None, 
                                layer_norm=True):
    
    return