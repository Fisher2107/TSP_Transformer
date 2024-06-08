import torch
import torch.nn as nn
import time
import argparse
from tqdm import tqdm

import os
import datetime

from torch.distributions.categorical import Categorical

# visualization 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try: 
    import networkx as nx
    from scipy.spatial.distance import pdist, squareform
    from concorde.tsp import TSPSolver # !pip install -e pyconcorde
except:
    pass
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from model import *

device = torch.device("cpu"); gpu_id = -1 # select CPU

#gpu_id = '0' # select a single GPU  
gpu_id = '0,1' # select multiple GPUs  
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU name: {:s}, gpu_id: {:s}'.format(torch.cuda.get_device_name(0),gpu_id))   
    
print(device)

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self
        
args = DotDict()
args.nb_nodes = 100 # TSP100
args.bsz = 512 #512 TSP20 TSP50
args.dim_emb = 128
args.dim_ff = 512
args.dim_input_nodes = 2
args.nb_layers_encoder = 6
args.nb_layers_decoder = 2
args.nb_heads = 8
args.nb_epochs = 10000
args.nb_batch_per_epoch = 250#2500
args.nb_batch_eval = 20
args.gpu_id = gpu_id
args.lr = 1e-4
args.tol = 1e-3
args.batchnorm = True  # if batchnorm=True  than batch norm is used
#args.batchnorm = False # if batchnorm=False than layer norm is used
args.max_len_PE = 1000

print(args)

###################
# Instantiate a training network and a baseline network
###################
try: 
    del model_train # remove existing model
    del model_baseline # remove existing model
except:
    pass

model_train = TSP_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, 
              args.nb_layers_encoder, args.nb_layers_decoder, args.nb_heads, args.max_len_PE,
              batchnorm=args.batchnorm)

model_baseline = TSP_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, 
              args.nb_layers_encoder, args.nb_layers_decoder, args.nb_heads, args.max_len_PE,
              batchnorm=args.batchnorm)

# uncomment these lines if trained with multiple GPUs
print(torch.cuda.device_count())
if torch.cuda.device_count()>1:
    model_train = nn.DataParallel(model_train)
    model_baseline = nn.DataParallel(model_baseline)
# uncomment these lines if trained with multiple GPUs

optimizer = torch.optim.Adam( model_train.parameters() , lr = args.lr ) 

model_train = model_train.to(device)
model_baseline = model_baseline.to(device)
model_baseline.eval()

print(args); print('')

# Logs
os.system("mkdir logs")
time_stamp=datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")
file_name = 'logs'+'/'+time_stamp + "-n{}".format(args.nb_nodes) + "-gpu{}".format(args.gpu_id) + ".txt"
file = open(file_name,"w",1) 
file.write(time_stamp+'\n\n') 
for arg in vars(args):
    file.write(arg)
    hyper_param_val="={}".format(getattr(args, arg))
    file.write(hyper_param_val)
    file.write('\n')
file.write('\n\n') 
plot_performance_train = []
plot_performance_baseline = []
all_strings = []
epoch_ckpt = 0
tot_time_ckpt = 0


###################
# Main training loop 
###################
start_training_time = time.time()
print('start train')
for epoch in tqdm(range(0,args.nb_epochs)):
    
    # re-start training with saved checkpoint
    epoch += epoch_ckpt

    ###################
    # Train model for one epoch
    ###################
    start = time.time()
    model_train.train() 
    for step in tqdm(range(1,args.nb_batch_per_epoch+1)):

        # generate a batch of random TSP instances    
        x = torch.rand(args.bsz, args.nb_nodes, args.dim_input_nodes, device=device) # size(x)=(bsz, nb_nodes, dim_input_nodes) 

        # compute tours for model
        tour_train, sumLogProbOfActions =  model_train(x, deterministic=False) # size(tour_train)=(bsz, nb_nodes), size(sumLogProbOfActions)=(bsz)
      
        # compute tours for baseline
        with torch.no_grad():
            tour_baseline, _ = model_baseline(x, deterministic=True)

        # get the lengths of the tours
        L_train = compute_tour_length(x, tour_train) # size(L_train)=(bsz)
        L_baseline = compute_tour_length(x, tour_baseline) # size(L_baseline)=(bsz)
        
        # backprop
        loss = torch.mean( (L_train - L_baseline)* sumLogProbOfActions )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    time_one_epoch = time.time()-start
    time_tot = time.time()-start_training_time + tot_time_ckpt

        
    ###################
    # Evaluate train model and baseline on 10k random TSP instances
    ###################
    model_train.eval()
    mean_tour_length_train = 0
    mean_tour_length_baseline = 0
    for step in range(0,args.nb_batch_eval):

        # generate a batch of random tsp instances   
        x = torch.rand(args.bsz, args.nb_nodes, args.dim_input_nodes, device=device) 

        # compute tour for model and baseline
        with torch.no_grad():
            tour_train, _ = model_train(x, deterministic=True)
            tour_baseline, _ = model_baseline(x, deterministic=True)
            
        # get the lengths of the tours
        L_train = compute_tour_length(x, tour_train)
        L_baseline = compute_tour_length(x, tour_baseline)

        # L_tr and L_bl are tensors of shape (bsz,). Compute the mean tour length
        mean_tour_length_train += L_train.mean().item()
        mean_tour_length_baseline += L_baseline.mean().item()

    mean_tour_length_train =  mean_tour_length_train/ args.nb_batch_eval
    mean_tour_length_baseline =  mean_tour_length_baseline/ args.nb_batch_eval

    # evaluate train model and baseline and update if train model is better
    update_baseline = mean_tour_length_train+args.tol < mean_tour_length_baseline
    if update_baseline:
        model_baseline.load_state_dict( model_train.state_dict() )


    # For checkpoint
    plot_performance_train.append([ (epoch+1), mean_tour_length_train])
    plot_performance_baseline.append([ (epoch+1), mean_tour_length_baseline])
        
    # Compute optimality gap
    if args.nb_nodes==50: gap_train = mean_tour_length_train/5.692- 1.0
    elif args.nb_nodes==100: gap_train = mean_tour_length_train/7.765- 1.0
    else: gap_train = -1.0
    
    # Print and save in txt file
    mystring_min = 'Epoch: {:d}, epoch time: {:.3f}min, tot time: {:.3f}day, L_train: {:.3f}, L_base: {:.3f}, L_test: {:.3f}, gap_train(%): {:.3f}, update: {}'.format(
        epoch, time_one_epoch/60, time_tot/86400, mean_tour_length_train, mean_tour_length_baseline, 100*gap_train, update_baseline) 
    print(mystring_min) # Comment if plot display
    file.write(mystring_min+'\n')
    
    # Saving checkpoint
    checkpoint_dir = os.path.join("checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save({
        'epoch': epoch,
        'time': time_one_epoch,
        'tot_time': time_tot,
        'loss': loss.item(),
        'TSP_length': [torch.mean(L_train).item(), torch.mean(L_baseline).item(), mean_tour_length_test],
        'plot_performance_train': plot_performance_train,
        'plot_performance_baseline': plot_performance_baseline,
        'model_baseline': model_baseline.state_dict(),
        'model_train': model_train.state_dict(),
        'optimizer': optimizer.state_dict(),
        }, '{}.pkl'.format(checkpoint_dir + "/checkpoint_" + time_stamp + "-n{}".format(args.nb_nodes) + "-gpu{}".format(args.gpu_id)))
