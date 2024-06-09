import os
import torch
import matplotlib.pyplot as plt
import sys
#read checkpoint
if "TSP" not in os.getcwd():
    os.chdir("TSP_Transformer")
checkpoint_file = sys.argv[1]
checkpoint = torch.load(checkpoint_file,map_location='cpu')
print('epoch =',checkpoint['epoch'])
print('time = ',checkpoint['time'], 'average = ', checkpoint['tot_time']/checkpoint['epoch'])
print('loss = ',checkpoint['loss'])
print('TSP_length = ',checkpoint['TSP_length'])
plt.plot([x[0] for x in checkpoint['plot_performance_train']], [x[1] for x in checkpoint['plot_performance_train']], label='train')
plt.show()