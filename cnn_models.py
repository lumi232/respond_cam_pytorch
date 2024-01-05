'''
Author: Guanan Zhao
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# CNN-1 in PyTorch
class CNN_1(nn.Module):
    def __init__(self, image_size, num_labels):
        super(CNN_1, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * image_size * image_size * image_size // 64, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_labels)

    def forward(self, x):
        x = F.max_pool3d(F.relu(self.conv1(x)), 2)
        x = F.max_pool3d(F.relu(self.conv2(x)), 2)
        x = F.max_pool3d(F.relu(self.conv3(x)), 2)
        x = F.max_pool3d(F.relu(self.conv4(x)), 2)
        x = x.view(x.size(0), -1) # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# CNN-2 in PyTorch
class CNN_2(CNN_1):
    def __init__(self, image_size, num_labels):
        super(CNN_2, self).__init__(image_size, num_labels)
        self.prepooling = nn.AvgPool3d(kernel_size=2, stride=None, padding=0)

    def forward(self, x):
        x = self.prepooling(x)
        return super(CNN_2, self).forward(x)

# Prediction function
def predict(model, data):
    model.eval()
    with torch.no_grad():
        data = torch.from_numpy(np.array(data)).float()
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        return predicted.numpy(), F.softmax(outputs, dim=1).numpy()

# Below are pre-processing functions
def vol_to_image_stack(vs):
    image_size = vs[0].shape[0]
    sample_size = len(vs)
    num_channels=1
    sample_data = N.zeros((sample_size, image_size, image_size, image_size, num_channels), dtype=N.float32)
    for i,v in enumerate(vs):
        sample_data[i, :, :, :, 0] = v
    return sample_data

def pdb_id_label_map(pdb_ids):
    pdb_ids = set(pdb_ids)
    pdb_ids = list(pdb_ids)
    m = {p:i for i,p in enumerate(pdb_ids)}
    return m

def list_to_data(dj, pdb_id_map=None):
    re = dict()
    re['data'] = vol_to_image_stack(vs=[_['v'] for _ in dj])

    if pdb_id_map is not None:
        labels = N.array([pdb_id_map[_['pdb_id']] for _ in dj])
        from keras.utils import np_utils
        labels = np_utils.to_categorical(labels, len(pdb_id_map))
        re['labels'] = labels
    return re
