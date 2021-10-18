import torch
import os
path = os.path.expanduser('~/models/vit-32.pt')
model = torch.load(path)
print(model['model_state_dict']['visual.conv1.weight'].size())
#print(len(model))
#for key in model.keys():
#    print(key)
#print(model)
