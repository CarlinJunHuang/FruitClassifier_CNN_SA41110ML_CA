import os, io, torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
import numpy as np


def print_named_params(model):
  for name, param in model.named_parameters():
    print(f"{name}: {param.numel()}")


def get_label_from_filename(filename):
    class_map = {"apple": 0, "banana": 1, "orange": 2, "mixed": 3}
    try:
        class_name = filename.split("_")[0].lower()
        return class_map.get(class_name, -1)
    except:
        return -1


def prepare_data(target_dir):
  filepaths = []
  labels = []

  # Scan single directory and parse filenames
  if os.path.exists(target_dir):
      files = os.listdir(target_dir)
      for file in files:
        if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
            
        label = get_label_from_filename(file)
        if label != -1:
            filepaths.append(f"{target_dir}/{file}")
            labels.append(label)
  else:
      print(f"Warning: Directory {target_dir} not found.")

  return np.array(filepaths), torch.tensor(labels)


def load_images(filepaths):
  # Instantiate class to transform image to tensor
  # Resize to 64x64 and ensure RGB
  to_tensor = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

  tensor = None

  # List all files in the directory
  for item in filepaths:
    image = Image.open(item).convert("RGB")
    #print(f"image size = {image.size}")

    # transforms.ToTensor() performs transformations on images
    # values of img_tensor are in the range of [0.0, 1.0]
    img_tensor = to_tensor(image) # convert into pytorch's tensor to work with
    #print(f"img_tensor.shape = {img_tensor.shape}")
    #input()

    if tensor is None:
      # size: [1,3,64,64] (Modified channels and size)
      tensor = img_tensor.unsqueeze(0) # add a new dimension
    else:
      # concatenate becomes [2,3,64,64], ...
      # dim=0 concatenates along the axis=0 (row-wise)
      tensor = torch.cat((tensor, img_tensor.unsqueeze(0)), dim=0)
    
  return tensor


class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()
    # in_channels=3 for RGB
    # out_channels=16 means we have 16 filters, each filter of size 3x3x3.
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    
    # in_channels=16 because our out_channels=16 from previous layer.
    # out_channels=32 means we are using 32 filters, each filter of size 3x3x16,
    # in this layer.
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    
    # Max Pooling Layer: downsample by a factor of 2.
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Fully Connected Layer 1: 
    # Input size calculation: 64x64 -> pool -> 32x32 -> pool -> 16x16.
    # Feature map size: 16 * 16 * 32.
    self.fc1 = nn.Linear(in_features= 7 * 7 * 32, out_features=128)
    
    # Fully Connected Layer 2: input size = 128, output size = 4 (for 4 output classes).
    self.fc2 = nn.Linear(in_features=128, out_features=4)

    # Activation function
    self.relu = nn.ReLU()

  def forward(self, x):
    #print(f"x.shape={x.shape}\n")

    # Apply convolution + ReLU + pooling
    x = self.conv1(x)
    x = self.relu(x)
    x = self.pool(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.pool(x)

    # Flatten the feature maps 
    x = x.view(-1, 7 * 7 * 32)

    # Fully connected layers
    x = self.fc1(x)
    x = self.relu(x)
    
    # Output layer (no activation since we apply softmax in the loss function)
    x = self.fc2(x)
    
    return x


def test(model, filepaths, labels):
  batch_size = 64
  samples_tested = 0
  correct_preds = 0
  total_samples = len(filepaths)

  for i in range(0, total_samples, batch_size):
    batch_inputs = load_images(filepaths[i : i + batch_size])
    batch_labels = labels[i : i + batch_size]
    
    if len(batch_inputs) == 0: continue

    # Forward pass: coyympute predicted outputs
    outputs = model(batch_inputs)

    # Get probability-distributions
    probs = torch.softmax(outputs, dim=1)
    _, preds = torch.max(probs, dim=1)

    # Determine accuracy
    samples_tested += len(batch_labels)
    correct_preds += torch.sum(preds == batch_labels)
    
  accuracy = correct_preds / float(samples_tested) if samples_tested > 0 else 0
  print(f"Test ({samples_tested}/{total_samples}): Accuracy={accuracy:.5f}")


def train(model, criterion, optimizer, filepaths, labels):
  # our hyper-parameters for training
  n_epochs = 2 
  batch_size = 64 

  for epoch in range(n_epochs):
    # For tracking and printing our training-progress
    samples_trained = 0
    run_loss = 0
    correct_preds = 0
    total_samples = len(filepaths) 

    permutation = torch.randperm(total_samples)
    for i in range(0, total_samples, batch_size):
      indices = permutation[i : i+batch_size]
      batch_inputs = load_images(filepaths[indices])
      batch_labels = labels[indices]
      
      if len(batch_inputs) == 0: continue

      # Forward pass: coyympute predicted outputs
      outputs = model(batch_inputs)

      # Compute loss
      loss = criterion(outputs, batch_labels)
      run_loss += loss.item()

      # Backward pass and optimization step
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      # Get probability-distributions
      probs = torch.softmax(outputs, dim=1)
      _, preds = torch.max(probs, dim=1)

      # Calculate some stats
      # samples_trained += len(indices)
      samples_trained += len(batch_labels)
      
      correct_preds += torch.sum(preds == batch_labels) # compare predictions with labels

    avg_loss = run_loss / total_samples  # Approximate average loss per batch
    accuracy = correct_preds / float(samples_trained) # cast to float to get "accuracy" in decimal 

    print(f"Epoch {epoch+1} " +
          f"({samples_trained}/{total_samples}): " +
          f"Loss={avg_loss:.5f}, Accuracy={accuracy:.5f}")



# Instantiate the model, define the loss function and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss() # define loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model 
# Modified for v1.1: Use data_relabeling
# Define paths (looking in parent directory if not found in current)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

dir_train = os.path.join(parent_dir, "data_relabeling")
dir_test = os.path.join(parent_dir, "test")

# Fallback
if not os.path.exists(dir_train): dir_train = "data_relabeling"
if not os.path.exists(dir_test): dir_test = "test"

filepaths_train, labels_train = prepare_data(dir_train)
filepaths_test, labels_test = prepare_data(dir_test)

if len(filepaths_train) > 0:
    print(f"Found {len(filepaths_train)} training images. Starting training...")
    train(model, criterion, optimizer, filepaths_train, labels_train)

    # Test the model
    if len(filepaths_test) > 0:
        print(f"Found {len(filepaths_test)} test images. Testing...")
        test(model, filepaths_test, labels_test)
    else:
        print("No test images found!")
else:
    print("No training images found. Please check the data directory.")
