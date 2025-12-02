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
  to_tensor = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

  tensor = None

  for item in filepaths:
    image = Image.open(item).convert("RGB")
    img_tensor = to_tensor(image)

    if tensor is None:
      tensor = img_tensor.unsqueeze(0)
    else:
      tensor = torch.cat((tensor, img_tensor.unsqueeze(0)), dim=0)
    
  return tensor


class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(in_features= 16 * 16 * 32, out_features=128)
    self.fc2 = nn.Linear(in_features=128, out_features=4)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.pool(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.pool(x)

    x = x.view(-1, 16 * 16 * 32)

    x = self.fc1(x)
    x = self.relu(x)
    
    x = self.fc2(x)
    
    return x


def test(model, filepaths, labels):
  batch_size = 64
  samples_tested = 0
  correct_preds = 0
  total_samples = len(filepaths)
  
  # Modified for v1.2: Track incorrect predictions
  incorrect_predictions = []
  ID_TO_CLASS = {0: "apple", 1: "banana", 2: "orange", 3: "mixed"}

  for i in range(0, total_samples, batch_size):
    batch_filepaths = filepaths[i : i + batch_size]
    batch_inputs = load_images(batch_filepaths)
    batch_labels = labels[i : i + batch_size]
    
    if len(batch_inputs) == 0: continue

    outputs = model(batch_inputs)

    probs = torch.softmax(outputs, dim=1)
    _, preds = torch.max(probs, dim=1)

    samples_tested += len(batch_labels)
    correct_preds += torch.sum(preds == batch_labels)
    
    # Modified for v1.2: Identify and store incorrect predictions
    wrong_mask = preds != batch_labels
    wrong_indices = torch.nonzero(wrong_mask, as_tuple=True)[0]
    
    for idx in wrong_indices:
        local_idx = idx.item()
        incorrect_predictions.append({
            "filename": os.path.basename(batch_filepaths[local_idx]),
            "true": ID_TO_CLASS[batch_labels[local_idx].item()],
            "pred": ID_TO_CLASS[preds[local_idx].item()]
        })
    
  accuracy = correct_preds / float(samples_tested) if samples_tested > 0 else 0
  print(f"Test ({samples_tested}/{total_samples}): Accuracy={accuracy:.5f}")
  
  # Modified for v1.2: Output incorrect predictions
  if incorrect_predictions:
      print("\nIncorrect Predictions:")
      for item in incorrect_predictions:
          print(f"  {item['filename']}: True={item['true']}, Pred={item['pred']}")
  else:
      print("\nNo incorrect predictions!")


def train(model, criterion, optimizer, filepaths, labels):
  n_epochs = 10 
  batch_size = 64 

  for epoch in range(n_epochs):
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

      outputs = model(batch_inputs)

      loss = criterion(outputs, batch_labels)
      run_loss += loss.item()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      probs = torch.softmax(outputs, dim=1)
      _, preds = torch.max(probs, dim=1)

      samples_trained += len(batch_labels)
      
      correct_preds += torch.sum(preds == batch_labels)

    avg_loss = run_loss / total_samples
    accuracy = correct_preds / float(samples_trained) 

    print(f"Epoch {epoch+1} " +
          f"({samples_trained}/{total_samples}): " +
          f"Loss={avg_loss:.5f}, Accuracy={accuracy:.5f}")



# Instantiate the model, define the loss function and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Modified for v1.2: Create outputs directory
os.makedirs("outputs", exist_ok=True)

# Modified for v1.2: Create outputs directory
os.makedirs("outputs", exist_ok=True)

# Define paths (looking in parent directory if not found in current)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

dir_train = os.path.join(parent_dir, "data_relabeling")
dir_test = os.path.join(parent_dir, "test")

# Fallback to current directory if parent paths don't exist (for robustness)
if not os.path.exists(dir_train):
    dir_train = "data_relabeling"
if not os.path.exists(dir_test):
    dir_test = "test"

print(f"Training data path: {dir_train}")
print(f"Test data path: {dir_test}")

# Train the model 
filepaths_train, labels_train = prepare_data(dir_train)
filepaths_test, labels_test = prepare_data(dir_test)

if len(filepaths_train) > 0:
    print(f"Found {len(filepaths_train)} training images. Starting training...")
    train(model, criterion, optimizer, filepaths_train, labels_train)
    
    # Modified for v1.2: Save model
    model_path = "outputs/model_v1.2.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    if len(filepaths_test) > 0:
        print(f"Found {len(filepaths_test)} test images. Testing...")
        test(model, filepaths_test, labels_test)
    else:
        print("No test images found!")
else:
    print("No training images found. Please check the data directory.")
