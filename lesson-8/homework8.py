# %%



import numpy as np
from PIL import Image


# %%
PREFIX = "https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle"
DATA_URL = f"{PREFIX}/hair_classifier_v1.onnx.data"
MODEL_URL = f"{PREFIX}/hair_classifier_v1.onnx"

import subprocess
import os

# Remove existing files if they exist
if os.path.exists("hair_classifier_v1.onnx.data"):
    os.remove("hair_classifier_v1.onnx.data")
if os.path.exists("hair_classifier_v1.onnx"):
    os.remove("hair_classifier_v1.onnx")
import subprocess
subprocess.run(["pip", "install", "pillow"])
subprocess.run(["pip", "install", "torch"])
subprocess.run(["pip", "install", "torchvision"])
subprocess.run(["wget", MODEL_URL])

# %%
# Answer1 = output check this again

# %%
# %pip install pillow
# %pip install torch
# %pip install torchvision

# %%
from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# %%
import requests

url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
r = requests.get(url)
r.raise_for_status()  # will raise an error if the download failed
subprocess.run(["pip", "show", "torchvision"])
with open("downloaded_image.jpeg", "wb") as f:
    f.write(r.content)


subprocess.run(["python", "-V"])
subprocess.run(["pip", "show", "torch", "torchvision"])

# %%

# %pip show torchvision



# %%
# %%python -V  # Removed invalid syntax
subprocess.run(["pip", "show", "torch", "torchvision"])


# %%
import torch
from torchvision import transforms

# Define the transformations
#
data_transforms = transforms.Compose([
    # 1. Resize the images to 200x200 pixels
    transforms.Resize((200, 200)),
    # 2. Convert the image (H, W, C) to a PyTorch Tensor (C, H, W) in [0, 1]
    transforms.ToTensor(),
    # 3. Normalize the tensor with mean and standard deviation for each channel
    # These are often standard values derived from large datasets like ImageNet,
    # or you can calculate them from your specific dataset.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %%
print(data_transforms)

# %%
# 1. Load and Resize (using PIL)
img = Image.open("downloaded_image.jpeg").convert('RGB')
img = img.resize((200, 200))  # Resize to 200x200 matches your model

# 2. Convert to NumPy Array
# Shape is currently (200, 200, 3) -> (Height, Width, Channels)
img_array = np.array(img)

# 3. Scale pixel values from [0, 255] to [0, 1]
img_array = img_array / 255.0

# 4. Normalize with Mean and Std (ImageNet Statistics)
# Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# We use broadcasting here. 
# (200, 200, 3) - (3,) works automatically in NumPy on the last dimension
img_array = (img_array - mean) / std

# 5. Transpose dimensions from HWC to CHW
# Current shape: (200, 200, 3) -> Desired shape: (3, 200, 200)
img_array = img_array.transpose(2, 0, 1)

# 6. Add Batch Dimension
# Shape becomes (1, 3, 200, 200)
img_array = np.expand_dims(img_array, axis=0)

# 7. Convert to Float32 (Standard for PyTorch models)
img_array = img_array.astype(np.float32)

print(f"Final shape: {img_array.shape}")
# You can now feed this into the model: model(torch.from_numpy(img_array))

# %%
import numpy as np
from PIL import Image

# 1. Load and Resize (using PIL)
img = Image.open("downloaded_image.jpeg").convert('RGB')
img = img.resize((200, 200))  # Resize to 200x200 matches your model

# 2. Convert to NumPy Array
# Shape is currently (200, 200, 3) -> (Height, Width, Channels)
img_array = np.array(img)

# 3. Scale pixel values from [0, 255] to [0, 1]
img_array = img_array / 255.0

# 4. Normalize with Mean and Std (ImageNet Statistics)
# Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# We use broadcasting here. 
# (200, 200, 3) - (3,) works automatically in NumPy on the last dimension
img_array = (img_array - mean) / std

# 5. Transpose dimensions from HWC to CHW
# Current shape: (200, 200, 3) -> Desired shape: (3, 200, 200)
img_array = img_array.transpose(2, 0, 1)

# 6. Add Batch Dimension
# Shape becomes (1, 3, 200, 200)
img_array = np.expand_dims(img_array, axis=0)

# 7. Convert to Float32 (Standard for PyTorch models)
img_array = img_array.astype(np.float32)

print(f"Final shape: {img_array.shape}")
# You can now feed this into the model: model(torch.from_numpy(img_array))

# %%

# Access the very first pixel's Red value
red_pixel_value = img_array[0, 0, 0, 0]

print(f"Red Channel Value (normalized): {red_pixel_value}")

# %%
# Question 3 Answer is -1.073 -> 2

# %%
# Apply the transform
# 'data_transforms' is the object defined earlier
input_tensor = data_transforms(img) 

# At this point, input_tensor shape is: (3, 200, 200)
# The model expects: (Batch_Size, 3, 200, 200)

# Add the batch dimension (unsqueeze)
input_batch = input_tensor.unsqueeze(0)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Model definition
class BinaryCNN(nn.Module):
    def __init__(self):
        super(BinaryCNN, self).__init__()
        # Input shape: (3, 200, 200)

        # 1. Convolutional Layer (nn.Conv2d)
        # Input: (3, 200, 200)
        # Output: 32 filters, kernel=(3,3), padding=0, stride=1
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            padding=0
        )
        # Activation is applied in the forward pass (F.relu)

        # 2. Max Pooling (nn.MaxPool2d)
        # Pooling size: (2, 2)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        # --- Calculate the size of the feature map after conv and pool ---
        # 200 - 3 + 1 = 198 (size after convolution)
        # 198 / 2 = 99 (size after max pooling)
        # Number of features to flatten: 32 channels * 99 * 99 = 313632

        # 3. Flatten (nn.Flatten)
        # This is handled in the forward pass using nn.Flatten() or .view()

        # 4. First Linear Layer (nn.Linear)
        self.fc1 = nn.Linear(32 * 99 * 99, 64)

        # 5. Output Linear Layer (nn.Linear)
        # Output: 1 neuron for binary classification
        self.fc2 = nn.Linear(64, 1)


    def forward(self, x):
        # Apply Conv layer and ReLU
        x = F.relu(self.conv1(x))

        # Apply Max Pooling
        x = self.pool(x)

        # Flatten the feature maps into a vector
        # x.shape will be (batch_size, 32, 99, 99)
        # Start flattening from dimension 1 (leaving batch size intact)
        x = torch.flatten(x, 1)

        # First Linear Layer and ReLU
        x = F.relu(self.fc1(x))

        # Return the raw logit (output of the last linear layer)
        output = self.fc2(x)

        return output

# Instantiate the model
model = BinaryCNN()
print("Model structure defined:")
print(model)

# %%
# Move to the same device as the model (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_batch = input_batch.to(device)

# Set model to evaluation mode (turns off training-specific features like Dropout)
model.eval()

with torch.no_grad():
    output = model(input_batch)
    
    # Apply sigmoid to get probability (since you used BCEWithLogitsLoss)
    probability = torch.sigmoid(output).item()

print(f"Probability of Class 1: {probability:.4f}")

# %%
# Question 4 Answer is 0.49 2


