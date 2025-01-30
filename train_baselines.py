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

from torch_geometric.utils import k_hop_subgraph
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

# MSE loss function
def msle_loss(pred, target):
    return mse_loss(torch.log(pred + 1),  torch.log(target + 1))

# scale loss function based on difference from L2
def weighted_mse_loss(pred, target, l2_diffs):
    errs = (target/l2_diffs) * torch.square(pred - target)
    return torch.mean(errs)

# NMSE loss function
def nmse_loss(pred, target):
    nz = torch.nonzero(target)
    errs = torch.square(pred[nz] - target[nz])/torch.square(target[nz])
    return torch.mean(errs)

def nmae_loss(pred, target):
    nz = torch.nonzero(target)
    errs = torch.abs(pred[nz] - target[nz])/torch.abs(target[nz])
    return torch.mean(errs)

def mse_loss(pred, target):
    return torch.mean(torch.square(pred - target))

def sqrt_distance(pred, target):
    sqrt_distance = torch.sqrt(target)
    return torch.mean(torch.square(pred - sqrt_distance))

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
    l2 = torch.norm(node_features[srcs] - node_features[tars], dim=1, p=2)

    train_dataset = SingleGraphShortestPathDataset(srcs, tars, lengths, l2)

    return train_dataset, node_features, edge_index


def train_single_graph_baseline1(node_features, edge_index, train_dataloader, 
                                 test_node_features, test_edge_index, test_dataloader, 
                                 model_config, layer_type, epochs=100, loss_func='nrmse_loss',
                                 device='cuda:0', log_dir='/data/sam',
                                 lr=0.001, siamese=True, save_freq=50, virtual_node=False, 
                                 aggr='sum', patch=False, metadata=None, p=1, hierarchical_vn_params=None,
                                 log=True, finetune=False, edge_attr = None, finetune_file=None):
    
    # initiate summary writer
    print(model_config)
    gnn_config = model_config['gnn']
    if not patch:
        edge_attr = torch.tensor(edge_attr) if edge_attr is not None else None
        edge_attr = edge_attr.unsqueeze(-1)
        graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        if log:
            test_graph = Data(x=test_node_features, edge_index=test_edge_index, edge_attr=edge_attr)
    
    # set edge dimensions
    if not patch:
        cnn_sz = int(np.sqrt(len(node_features)))

        if layer_type=='CNNLayer':
            feats = graph_data.x.numpy()
            total_num_nodes = len(feats)
            cnn_img =  feats.T.reshape(3, -1, total_num_nodes).swapaxes(0, 1).reshape(1, 3, cnn_sz, cnn_sz)
            cnn_in = torch.tensor(cnn_img, device=device)
    
    edge_dim = 1 if edge_attr is not None else 0
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
    elif aggr == 'sum+diff+vn' and layer_type != 'CNNLayer':
        gnn_model = GNN_Final_VN_Model(batches=patch, layer_type=layer_type, edge_dim=edge_dim, **gnn_config)
    elif aggr  == 'sum+diff+vn' and layer_type == 'CNNLayer':
        gnn_model = CNN_Final_VN_Model(batches=patch, layer_type=layer_type, edge_dim=edge_dim, **gnn_config)
    # Transformer
    elif layer_type == 'Transformer':
        gnn_model = Transformer(**gnn_config)
        #gnn_model = GraphTransformer(**gnn_config)
    # Add laplacian positional encodings to the transformer (k = 10 by default)
    elif layer_type == 'Transformer-LPE':
        print("Laplacian positional encodings")
        graph_data = add_laplace_positional_encoding(graph_data, k=10)
        test_graph = add_laplace_positional_encoding(test_graph, k=10)
        gnn_config['input'] = graph_data.x.size()[1]
        gnn_model=GraphTransformer(**gnn_config)
    elif layer_type == "MLP":
        deepsets_config = model_config['gnn']
        gnn_model = initialize_mlp(**deepsets_config, activation='lrelu')
    else:
        print("Train vanilla GNN")
        gnn_model = GNNModel(layer_type=layer_type, edge_dim=edge_dim, **gnn_config)
    
    print("Sending model to GPU.....")
    gnn_model = gnn_model.to(torch.double)
    gnn_model.to(device)

    mlp=None
    if siamese and not finetune:
        parameters = gnn_model.parameters()
        
    else:
        mlp_config = model_config['mlp']
        print(mlp_config)
        if aggr == 'combine':
            mlp_config['input'] = mlp_config['input'] * 3
        elif aggr == 'concat':
            mlp_config['input'] = mlp_config['input'] * 2 
        elif aggr == 'sum+diff':
            mlp_config['input'] = gnn_config['output'] * 2
        elif aggr == 'sum+diff+vn':
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
        if not siamese:
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
        
        
        parameters = mlp.parameters()
        for param in gnn_model.parameters():
            param.requires_grad =False
        siamese = False
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
    node_embeddings = None
    vn_emb= None
    if finetune:
        gnn_model.to('cpu')
        node_embeddings = gnn_model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        node_embeddings = node_embeddings.to(device)
        torch.save(node_embeddings,'/data/sam/terrain/data/norway/norway-2k-embeddings.pt')
    elif not patch:
        graph_data = graph_data.to(device)
    
    print(finetune == False or (finetune and node_embeddings == None))
    optimizer.zero_grad()
    
    for epoch in trange(epochs):
        total_loss = 0
        batch_count = 0
        times = []
        for batch in train_dataloader:
            
            if patch:
                batch.to(device)
            else:
                srcs = batch[0].to(device)
                tars = batch[1].to(device)
                lengths = batch[2].to(device)
                l2 = batch[3].to(device)

            # node_features = node_features.to(device)

            # format patch data for CNN Layers
            if layer_type == 'CNNLayer' and patch:
                batch_size = batch.num_graphs
                batch.to(device)
            
            # edge_index = edge_index.to(device)
            
            if finetune == False or (finetune and node_embeddings == None):
                if aggr == 'sum+diff+vn' and layer_type != 'CNNLayer':
                    if patch:
                        node_embeddings, vn_emb = gnn_model(batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch)
                    else:
                        node_embeddings, vn_emb =gnn_model(graph_data.x, graph_data.edge_index, edge_attr = graph_data.edge_attr)
                elif aggr == 'sum+diff+vn' and layer_type == 'CNNLayer':
                    if patch:
                        graph_sz = int(np.sqrt(len(batch.x)/batch.num_graphs))
                        cnn_input = batch.x.reshape(batch.num_graphs, graph_sz * graph_sz, 3).mT.reshape(batch.num_graphs, 3, graph_sz, graph_sz)
                        node_embeddings, vn_emb = gnn_model(cnn_input, batch.edge_index, edge_attr=batch.edge_attr, batch=batch)
                    else:
                        node_embeddings, vn_emb =gnn_model(cnn_in, edge_index=None)
                elif virtual_node:
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
                    if not patch:
                        node_embeddings = gnn_model(cnn_in, edge_index = None)
                    else:
                        graph_sz = int(np.sqrt(len(batch.x)/batch.num_graphs))
                        cnn_input = batch.x.reshape(batch.num_graphs, graph_sz * graph_sz, 3).mT.reshape(batch.num_graphs, 3, graph_sz, graph_sz)
                        node_embeddings = gnn_model(cnn_input, edge_index=None, batch=batch.num_graphs)
                elif layer_type == 'Transformer':
                    transformer_input = graph_data.x.unsqueeze(dim=0)
                    node_embeddings = gnn_model(transformer_input, edge_index=None)
                else:
                    node_embeddings = gnn_model(graph_data.x, graph_data.edge_index, edge_attr = graph_data.edge_attr)

                    #node_embeddings = gnn_model(batch.x, batch.edge_index, edge_attr=batch.edge_attr) if patch else gnn_model(graph_data.x, graph_data.edge_index, edge_attr = graph_data.edge_attr)
            if siamese:
                pred = torch.norm(node_embeddings[batch.src] - node_embeddings[batch.tar], p=p, dim=1) if patch else torch.norm(node_embeddings[srcs] - node_embeddings[tars], p=p, dim=1)
            else:
                if not patch:
                    pred = mlp(node_embeddings[srcs], node_embeddings[tars], vn_emb=vn_emb)
                else:
                    pred = mlp(node_embeddings[batch.src], node_embeddings[batch.tar], vn_emb=vn_emb, batch=patch)
                pred = pred.squeeze()

            #loss = nrmse_loss(pred, lengths)
            if loss_func == 'weighted_mse_loss':
                loss = globals()[loss_func](pred, lengths,l2 )
                #print("time to compute loss:", end - start)
            else:
                loss = globals()[loss_func](pred, batch.length) if patch else globals()[loss_func](pred, lengths)
            total_loss += loss.detach()
            batch_count += 1
            loss.backward()

            optimizer.step()

            optimizer.zero_grad()
        writer.add_scalar('train/mse_loss', total_loss/batch_count, epoch)
        # print(epoch,total_loss/batch_count )
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
    print("Final training loss:", total_loss/batch_count)
    logging.info(f'final training loss: {total_loss/batch_count}')
    if siamese:
        path = os.path.join(log_dir, 'final_model.pt')
        print("saving model to:", path)
        torch.save(gnn_model.state_dict(), path)
        return gnn_model
    else:
        path = os.path.join(log_dir, 'final_model.pt')
        torch.save({'gnn_state_dict':gnn_model.state_dict(), 
                    'mlp_state_dict': mlp.state_dict()}, path)
        return gnn_model, mlp



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