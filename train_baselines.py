import argparse
from tqdm import tqdm, trange
from torch.optim import Adam
from torch_geometric.data import Data

from baselines import *
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.transforms import ToSparseTensor, VirtualNode
from transforms import add_laplace_positional_encoding
import yaml
import os
import csv

mse_loss = nn.MSELoss()

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
    
def compute_validation_loss(node_embeddings, test_dataloader, mlp=None, device='cuda:0'):
    total = 0
    count = 0
    for batch in test_dataloader:
        srcs = batch[0].to(device)
        tars = batch[1].to(device)
        lengths = batch[2].to(device)
        if mlp == None:
            pred = torch.norm(node_embeddings[srcs] - node_embeddings[tars], p=2, dim=1)
        else: 
            pred = mlp(node_embeddings[srcs], node_embeddings[tars])
        relative_losses = torch.sum(torch.abs(pred - lengths)/lengths)
        total += relative_losses.detach()
        count += len(srcs)

    return total/count

def npz_to_dataset(data):
    
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long)

    srcs = torch.tensor(data['srcs'])
    tars = torch.tensor(data['tars'])
    lengths = torch.tensor(data['lengths'])
    node_features = torch.tensor(data['node_features'], dtype=torch.double)

    train_dataset = SingleGraphShortestPathDataset(srcs, tars, lengths)

    return train_dataset, node_features, edge_index

def train_single_graph_baseline1(node_features, edge_index, train_dataloader, 
                                 test_node_features, test_edge_index, test_dataloader, 
                                 model_config, epochs=100, 
                                 device='cuda:0', log_dir='/data/sam',
                                 lr=0.001, siamese=True, save_freq=50, virtual_node=False):
    
    # initiate summary writer
    
    gnn_config = model_config['gnn']
    graph_data = Data(x=node_features, edge_index=edge_index)
    test_graph = Data(x=test_node_features, edge_index=test_edge_index)
    # Add virtual node
    if virtual_node:
        graph_data = virtual_node_transform(graph_data)

    if gnn_config['layer_type'] == 'GCN2Conv':
        gnn_model = GCN2Model(**gnn_config)
        graph_data = sparse_tensor(graph_data)
        test_graph = sparse_tensor(test_graph)
    elif gnn_config['layer_type'] == 'Transformer':
        gnn_model = GraphTransformer(**gnn_config)
    elif gnn_config['layer_type'] == 'Transformer-LPE':
        print("Laplacian positional encodings")
        graph_data = add_laplace_positional_encoding(graph_data, k=4)
        test_graph = add_laplace_positional_encoding(test_graph, k=4)
        gnn_config['input'] = graph_data.x.size()[1]
        gnn_model=GraphTransformer(**gnn_config)
    else:
        print("Train GNN")
        gnn_model = GNNModel(**gnn_config)
    gnn_model = gnn_model.to(torch.double)

    mlp=None

    gnn_model.to(device)
    if siamese:
        parameters = gnn_model.parameters()
    else:
        phi_config = model_config['phi']
        phi = initialize_mlp(**phi_config)
        final_config = model_config['final']
        final = initialize_mlp(**final_config)
        mlp = MLPBaseline0(phi, final)
        mlp.init_weights()
        mlp = mlp.to(torch.double)
        mlp.to(device)

        parameters = list(gnn_model.parameters()) + list(mlp.parameters())
    
    optimizer = Adam(parameters, lr=lr)

    record_dir = os.path.join(log_dir, 'record/')
    
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    print("logging losses and intermediary models to:", record_dir)
    writer = SummaryWriter(log_dir=record_dir)
    for epoch in trange(epochs):
        total_loss = 0
        batch_count = 0
        for batch in train_dataloader:
            batch_count += 1

            optimizer.zero_grad()
            
            srcs = batch[0].to(device)
            tars = batch[1].to(device)
            lengths = batch[2].to(device)
            graph_data = graph_data.to(device)
            # node_features = node_features.to(device)
            
            # edge_index = edge_index.to(device)
            if gnn_config['layer_type'] == 'GCN2Conv':
                node_embeddings = gnn_model(graph_data.x, graph_data.adj_t)
            else:
                node_embeddings = gnn_model(graph_data.x, graph_data.edge_index)
            if siamese:
                pred = torch.norm(node_embeddings[srcs] - node_embeddings[tars], p=2, dim=1)
            else:
                pred = mlp(node_embeddings[srcs], node_embeddings[tars])
                pred = pred.squeeze()

            loss = mse_loss(pred, lengths)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
        test_graph = test_graph.to(device)
        # test_node_features = test_node_features.to(device)
        # test_edge_index = test_edge_index.to(device)
        if gnn_config['layer_type'] == 'GCN2Conv':
            node_embeddings = gnn_model(test_graph.x, test_graph.adj_t)
        else:
            node_embeddings = gnn_model(test_graph.x, test_graph.edge_index)
        val_loss = compute_validation_loss(node_embeddings, test_dataloader, mlp=mlp, device=device)
        train_relative_loss = compute_validation_loss(node_embeddings, train_dataloader, mlp=mlp, device=device)
        if epoch == epochs - 1:
            print("epoch:", epoch, "test loss (relative error):", val_loss)

        writer.add_scalar('train/mse_loss', total_loss/batch_count, epoch)
        writer.add_scalar('train/relative_loss', train_relative_loss, epoch)
        writer.add_scalar('test/relative_loss', val_loss, epoch)
        if epoch % save_freq == 0:
            if siamese:
                path = os.path.join(record_dir, f'model_{epoch}.pt')
                torch.save({'gnn_state_dict': gnn_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, path)
            else:
                path = os.path.join(record_dir, f'model_{epoch}.pt')
                torch.save({'gnn_state_dict':gnn_model.state_dict(), 
                            'mlp_state_dict': mlp.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict()}, path)


    if siamese:
        path = os.path.join(log_dir, 'final_model.pt')
        print("saving model to:", path)
        torch.save(gnn_model.state_dict(), path)
        return gnn_model, val_loss
    else:
        path = os.path.join(log_dir, 'final_model.pt')
        torch.save({'gnn_state_dict':gnn_model.state_dict(), 
                    'mlp_state_dict': mlp.state_dict()}, path)
        return gnn_model, mlp, val_loss

def parse_config(filename):
    name = filename.split("/")[1]
    modeltype = name.split('-')[2].split('.')[0]
    return modeltype

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, help='Needed to construct output directory')
    parser.add_argument('--train-data', type=str)
    parser.add_argument('--test-data', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--config', type=str, default='configs/config-base.yml')
    parser.add_argument('--device', type=str)
    parser.add_argument('--siamese', type=int, default=0)
    parser.add_argument('--vn', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)

    args = parser.parse_args()
    siamese = True if args.siamese == 1 else False
    vn = True if args.vn == 1 else False 

    # Load data 
    train_file = os.path.join(output_dir, 'data', args.train_data)
    test_file = os.path.join(output_dir, 'data', args.train_data)

    train_data = np.load(train_file, allow_pickle=True)
    test_data = np.load(test_file, allow_pickle=True)

    train_dataset, train_node_features, train_edge_index = npz_to_dataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset, test_node_features, test_edge_index = npz_to_dataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=True)

    # Load model configs
    with open(args.config, 'r') as file:
        model_configs = yaml.safe_load(file)
    
    loss_data = []
    
    for modelname in model_configs:
        # Format log directory
        if siamese:
            log_dir = os.path.join(output_dir, 'models', args.dataset_name, 'baseline1/siamese', modelname, 'vn' if vn else 'no-vn')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        else:
            log_dir = os.path.join(output_dir, 'models', args.dataset_name, 'baseline1/mlp', modelname, 'vn' if vn else 'no-vn')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        config=model_configs[modelname]

        output = train_single_graph_baseline1(train_node_features, train_edge_index, train_dataloader, 
                                            test_node_features, test_edge_index, test_dataloader, 
                                            model_config = config, epochs=args.epochs, device=args.device,
                                            siamese=siamese, log_dir=log_dir, virtual_node=vn)
        loss_data.append({'modelname':modelname, 'loss':output[-1]})

    # Keep track of validation losses for each configuration
    if 'base' in args.config:
        print("finished training base model")
        return 0
    modeltype = parse_config(args.config)
    fieldnames = ['modelname', 'loss']
    csv_file = os.path.join('output', args.dataset_name, 'baseline1', 'siamese' if siamese else 'mlp', 'vn' if vn else 'no-vn')
    if not os.path.exists(csv_file):
        os.makedirs(csv_file)
    csv_file = csv_file + f'/{modeltype}.csv'
    csvfile = open(csv_file, 'w', newline='')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in loss_data:
        writer.writerow(row)
    csvfile.close()
    print(f'Data has been written to {csv_file}')
    
if __name__=='__main__':
    main()