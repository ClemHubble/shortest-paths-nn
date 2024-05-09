import argparse
from tqdm import tqdm, trange
from torch.optim import Adam, AdamW
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader

from src.baselines import *
import numpy as np
import torch
import torch.nn as nn
from torchmetrics.regression import MeanAbsolutePercentageError

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.transforms import ToSparseTensor, VirtualNode, ToUndirected
from torch_geometric.nn import to_hetero
from src.transforms import add_laplace_positional_encoding, add_virtual_node
import yaml
import os
import csv
import logging

import time

from torch.profiler import profile, record_function, ProfilerActivity

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

# MSE loss function
def msle_loss(pred, target):
    return mse_loss(torch.log(pred + 1),  torch.log(target + 1))

# NMSE loss function
def nmse_loss(pred, target):
    nz = torch.nonzero(target)
    errs = torch.square(pred[nz] - target[nz])/torch.square(target[nz])
    return torch.mean(errs)

def mse_loss(pred, target):
    return torch.mean(torch.square(pred - target))

def compute_validation_loss(node_embeddings, dataloader, mlp=None, device='cuda:0'):
    total = 0
    count = 0
    for batch in dataloader:
        srcs = batch[0].to(device)
        tars = batch[1].to(device)
        lengths = batch[2].to(device)
        if mlp == None:
            pred = torch.norm(node_embeddings[srcs] - node_embeddings[tars], p=1, dim=1)
        else: 
            pred = mlp(node_embeddings[srcs], node_embeddings[tars])
            pred = pred.squeeze()
        nz = torch.nonzero(lengths)
        relative_losses = torch.sum(torch.abs(pred[nz] - lengths[nz])/lengths[nz])
        total += relative_losses.detach()
        count += len(srcs)

    return total/count

def compute_patch_validation_loss(gnn_model, dataloader, mlp=None, device='cuda:0'):
    total = 0
    count = 0
    for batch in dataloader:
        batch = batch.to(device)
        if isinstance(gnn_model, GNNModel):
            node_embeddings = gnn_model(batch.x, batch.edge_index)
        elif isinstance(gnn_model, VNModel):
            node_embeddings = gnn_model(batch)
        else:
            node_embeddings = gnn_model(batch.x, batch.edge_index, batch=batch)
        if mlp == None:
            pred = torch.norm(node_embeddings[batch.src] - node_embeddings[batch.tar], p=1, dim=1)
        else:
            pred = mlp(node_embeddings[batch.src], node_embeddings[batch.tar])
            pred=pred.squeeze()
        nz = torch.nonzero(batch.length)
        if len(batch.length) == 1 and len(nz) == 1:
            continue
        elif len(batch.length) == 1:
            relative_losses = torch.sum(torch.abs(pred - batch.length)/batch.length)
        else:
            relative_losses = torch.sum(torch.abs(pred[nz] - batch.length[nz])/batch.length[nz])
        total += relative_losses.detach()
        count += len(batch.src)

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
                                 model_config, layer_type, epochs=100, loss_func='nrmse_loss',
                                 device='cuda:0', log_dir='/data/sam',
                                 lr=0.001, siamese=True, save_freq=50, virtual_node=False, 
                                 aggr='sum', patch=False, metadata=None, p=1, hierarchical_vn_params=None,
                                 log=True, finetune=False, edge_attr = None, finetune_file=None):
    
    # initiate summary writer
    
    gnn_config = model_config['gnn']
    if not patch:
        edge_attr = torch.tensor(edge_attr) if edge_attr is not None else None
        graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        if log:
            test_graph = Data(x=test_node_features, edge_index=test_edge_index, edge_attr=edge_attr)
    
    # set edge dimensions
    print(type(graph_data.edge_attr))
    cnn_sz = int(np.sqrt(len(node_features)))

    if layer_type=='CNNLayer':
        feats = graph_data.x.numpy()
        total_num_nodes = len(feats)
        cnn_img =  feats.T.reshape(3, -1, total_num_nodes).swapaxes(0, 1).reshape(1, 3, cnn_sz, cnn_sz)
        cnn_in = torch.tensor(cnn_img, device=device)
    
    edge_dim = 1 if isinstance(graph_data.edge_attr, torch.Tensor) else 0
    # Add virtual node
    if virtual_node:
        gnn_model = GNN_VN_Model(batches=patch, layer_type=layer_type, edge_dim=edge_dim, **gnn_config)
    elif hierarchical_vn_params != None:
        gnn_model = GNN_VN_Hierarchical(layer_type=layer_type, **gnn_config)

        # initialize necessary support for hierarchical virtual node message passing
        # hierarchical_vn_params = {n: 100, m = 100, pct_n: 0.2, pct_m: 0.2 , h_layers: 1}
        sz_n = int(hierarchical_vn_params['n'] * hierarchical_vn_params['pct_n'])
        sz_m = int(hierarchical_vn_params['m'] * hierarchical_vn_params['pct_m'])
        hblocks = None
        hnum = None
        hlayers = 0
    # GCN2Conv 
    elif layer_type == 'GCN2Conv':
        gnn_model = GCN2Model(**gnn_config)
        graph_data = sparse_tensor(graph_data)
        test_graph = sparse_tensor(test_graph)
    # Transformer
    elif layer_type == 'Transformer':
        gnn_model = GraphTransformer(**gnn_config)
    # Add laplacian positional encodings to the transformer (k = 10 by default)
    elif layer_type == 'Transformer-LPE':
        print("Laplacian positional encodings")
        graph_data = add_laplace_positional_encoding(graph_data, k=10)
        test_graph = add_laplace_positional_encoding(test_graph, k=10)
        gnn_config['input'] = graph_data.x.size()[1]
        gnn_model=GraphTransformer(**gnn_config)
    elif layer_type == "MLP":
        deepsets_config = model_config['gnn']
        gnn_model = initialize_mlp(**deepsets_config)
    else:
        print("Train vanilla GNN")
        gnn_model = GNNModel(layer_type=layer_type, edge_dim=edge_dim, size=cnn_sz, **gnn_config)
    
    print("Sending model to GPU.....")
    gnn_model = gnn_model.to(torch.double)
    gnn_model.to(device)

    mlp=None
    if siamese:
        parameters = gnn_model.parameters()
        
    else:
        mlp_config = model_config['mlp']
        if aggr == 'combine':
            mlp_config['input'] = mlp_config['input'] * 3
        mlp_nn = initialize_mlp(**mlp_config, activation='lrelu')
        mlp = MLPBaseline1(mlp_nn, aggr=aggr)
        mlp.init_weights()
        mlp = mlp.to(torch.double)
        mlp.to(device)

        parameters = list(gnn_model.parameters()) + list(mlp.parameters())
        
    
    
    # if we finetune the model, we freeze the siamese layer and just train the MLP
    if finetune:
        # load previous model from log
        
        prev_model_pth = os.path.join(log_dir, 'final_model.pt')

        print("finetuning from file:", prev_model_pth)
        model_info = torch.load(prev_model_pth, map_location=device)
        if mlp:
            mlp_parameters = model_info['mlp_state_dict']
            gnn_parameters = model_info['gnn_state_dict']
            
            mlp.load_state_dict(mlp_parameters)
            gnn_model.load_state_dict(gnn_parameters)
            #parameters = list(gnn_model.parameters()) + list(mlp.parameters())

        else:
            gnn_model.load_state_dict(model_info)
            #parameters = gnn_model.parameters()
        if finetune_file is not None:
            log_dir = finetune_file
        else:
            log_dir = os.path.join(log_dir, 'finetune/')
        
        
        # parameters = mlp.parameters()
        # for param in gnn_model.parameters():
        #     param.requires_grad =False
        # siamese = False
    record_dir = os.path.join(log_dir, 'record/')

    optimizer = AdamW(parameters, lr=lr)
    
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    print("logging losses and intermediary models to:", record_dir)
    writer = SummaryWriter(log_dir=record_dir)

    # initialize logger
    log_file = os.path.join(record_dir, 'training_log.log')
    logging.basicConfig(level=logging.INFO, filename=log_file)

    logging.info(f'GNN layer: {layer_type}')
    logging.info(f'Number of epochs: {epochs}')
    logging.info(f'MLP aggregation: {aggr}')
    logging.info(f'Siamese? {siamese}')
    logging.info(f'loss function: {loss_func}')
    logging.info(f'edge attributes?: {edge_attr != None}')

    logging.info(gnn_model)
    logging.info(mlp)
    
    logging.info('training......')
    start = time.time()
    for epoch in trange(epochs):
        total_loss = 0
        batch_count = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
                # print(gnn_model(cnn_in, edge_index=None).size())
                # h()
            if patch:
                batch.to(device)
            else:
                srcs = batch[0].to(device)
                tars = batch[1].to(device)
                lengths = batch[2].to(device)
                graph_data = graph_data.to(device)
            # node_features = node_features.to(device)
            
            # edge_index = edge_index.to(device)
            if layer_type == 'GCN2Conv' or virtual_node:
                # node_embeddings = gnn_model(graph_data)
                node_embeddings = gnn_model(batch.x , batch.edge_index, edge_attr=batch.edge_attr, batch=batch) if patch else gnn_model(graph_data.x, graph_data.edge_index, edge_attr = graph_data.edge_attr)
            elif layer_type == 'MLP':
                # node_embeddings = gnn_model(graph_data.x)
                node_embeddings = gnn_model(batch.x) if patch else gnn_model(graph_data.x)
            elif isinstance(gnn_model, GNN_VN_Hierarchical):
                if patch:
                    raise Exception("Not supported for patch datasets")
                node_embeddings = gnn_model(graph_data.x, graph_data.edge_index, hblocks, 1, hnum)
            elif layer_type=='CNNLayer':
                node_embeddings = gnn_model(cnn_in, edge_index = None)
            else:
                # node_embeddings = gnn_model(graph_data.x, graph_data.edge_index)
                node_embeddings = gnn_model(batch.x, batch.edge_index, edge_attr=batch.edge_attr) if patch else gnn_model(graph_data.x, graph_data.edge_index, edge_attr = graph_data.edge_attr)

            if siamese:
                # pred = torch.norm(node_embeddings[srcs] - node_embeddings[tars], p=1, dim=1)
                pred = torch.norm(node_embeddings[batch.src] - node_embeddings[batch.tar], p=p, dim=1) if patch else torch.norm(node_embeddings[srcs] - node_embeddings[tars], p=p, dim=1)
            else:
                pred = mlp(node_embeddings[batch.src], node_embeddings[batch.tar]) if patch else mlp(node_embeddings[srcs], node_embeddings[tars])
                pred = pred.squeeze()

            #loss = nrmse_loss(pred, lengths)
            loss = globals()[loss_func](pred, batch.length) if patch else globals()[loss_func](pred, lengths)
            total_loss += loss.detach()
            batch_count += 1
            loss.backward()
            optimizer.step()

        if log:
            if not patch:
                test_graph = test_graph.to(device)
                graph_data = graph_data.to(device)

                if layer_type == 'GCN2Conv':
                    test_node_embeddings = gnn_model(test_graph)
                    train_node_embeddings = gnn_model(graph_data)
                elif layer_type=='MLP':
                    test_node_embeddings = gnn_model(test_graph.x)
                    train_node_embeddings = gnn_model(graph_data.x)
                elif isinstance(gnn_model, GNN_VN_Hierarchical):
                    test_node_embeddings = gnn_model(test_graph.x, test_graph.edge_index, hblocks, 1, hnum)
                    train_node_embeddings = gnn_model(graph_data.x, graph_data.edge_index, hblocks, 1, hnum)
                elif layer_type == 'CNNLayer':
                    test_node_embeddings = gnn_model(cnn_in, edge_index = None)
                    train_node_embeddings = gnn_model(cnn_in, edge_index = None)
                else:
                    test_node_embeddings = gnn_model(test_graph.x, test_graph.edge_index)
                    train_node_embeddings = gnn_model(graph_data.x, graph_data.edge_index)

                val_loss = compute_validation_loss(test_node_embeddings, test_dataloader, mlp=mlp, device=device)
                train_relative_loss = compute_validation_loss(train_node_embeddings, train_dataloader, mlp=mlp, device=device)
            else:
                val_loss = compute_patch_validation_loss(gnn_model, test_dataloader, mlp=mlp, device=device)
                train_relative_loss = compute_patch_validation_loss(gnn_model, train_dataloader, mlp=mlp, device=device)
                
            if epoch == epochs - 1:
                end = time.time()
                logging.info(f'Trained in {end - start} seconds')
                logging.info(f"epoch:{epoch} train loss:{(total_loss/batch_count).item()}")
                logging.info(f"epoch:{epoch} test loss (relative error):{val_loss.item()}")
                print("epoch:", epoch, "test loss (relative error):", val_loss)
                print("epoch:", epoch, "train loss", train_relative_loss)
                print("epoch:", epoch, "total epoch loss", total_loss/batch_count)
                
            
            writer.add_scalar('train/mse_loss', total_loss/batch_count, epoch)
            writer.add_scalar('train/relative_loss', train_relative_loss, epoch)
            writer.add_scalar('test/relative_loss', val_loss, epoch)
            writer.add_scalars('test/generalization-relative', 
                            {'val': val_loss, 
                            'train': train_relative_loss}, 
                            epoch)
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


def format_log_dir(output_dir, dataset_name, siamese, modelname, vn, max, loss_func):
    log_dir = os.path.join(output_dir, 
                           'models', 
                           dataset_name, 
                           'baseline1/siamese' if siamese else 'baseline1/mlp', 
                           modelname, 
                           'vn' if vn else 'no-vn')
    if not siamese:
        log_dir = os.path.join(log_dir, 'max' if max else 'sum')
    log_dir = os.path.join(log_dir, loss_func)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def parse_config(filename):
    name = filename.split("/")[2]
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
    parser.add_argument('--max', type=int, default=1)
    parser.add_argument('--loss', type=str, default='mse_loss')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--finetune', type=int, default=0)

    args = parser.parse_args()
    siamese = True if args.siamese == 1 else False
    vn = True if args.vn == 1 else False 
    max_agg = True if args.max == 1 else False

    finetune= True if args.finetune==1 else False

    # Load data 
    train_file = os.path.join(output_dir, 'data', args.train_data)
    test_file = os.path.join(output_dir, 'data', args.train_data)

    train_data = np.load(train_file, allow_pickle=True)
    test_data = np.load(test_file, allow_pickle=True)

    train_dataset, train_node_features, train_edge_index = npz_to_dataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset, test_node_features, test_edge_index = npz_to_dataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False)

    # Load model configs
    with open(args.config, 'r') as file:
        model_configs = yaml.safe_load(file)
    
    loss_data = []
    
    for modelname in model_configs:
        log_dir = format_log_dir(output_dir, args.dataset_name, siamese, modelname, vn, max_agg, args.loss)
        config=model_configs[modelname]

        output = train_single_graph_baseline1(train_node_features, train_edge_index, train_dataloader, 
                                            test_node_features, test_edge_index, test_dataloader, loss_func=args.loss,
                                            model_config = config, epochs=args.epochs, device=args.device,
                                            siamese=siamese, log_dir=log_dir, virtual_node=vn, max=max_agg, lr=args.lr, p=args.p,
                                            log=False, finetune=finetune)
        loss_data.append({'modelname':modelname, 'loss':output[-1]})

    # Keep track of validation losses for each configuration
    if 'base' in args.config:
        print("finished training base model")
        return 0
    modeltype = parse_config(args.config)
    fieldnames = ['modelname', 'loss']
    csv_file = os.path.join('output', 
                            args.dataset_name, 
                            'baseline1', 
                            'siamese' if siamese else 'mlp', 
                            'vn' if vn else 'no-vn', 
                            'max' if max_agg else 'sum', 
                            f'p-{args.p}',
                            args.loss)
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