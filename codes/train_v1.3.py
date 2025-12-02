import os
import io
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import datetime

# ==========================================
# Global Hyperparameters & Settings
# ==========================================
TEAM_MEMBER_NAME = "Jun"       # Name of the team member running the experiment
TRIAL_PARAM_NAME = "Baseline"  # Parameter being tested (e.g., "lr_0.01", "epoch_20")
NUM_EPOCHS = 10
BATCH_SIZE = 32
IMAGE_SIZE = 32
LEARNING_RATE = 0.001

# Class mapping
ID_TO_CLASS = {0: "apple", 1: "banana", 2: "orange", 3: "mixed"}

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
                filepaths.append(os.path.join(target_dir, file))
                labels.append(label)
    else:
        print(f"Warning: Directory {target_dir} not found.")

    return np.array(filepaths), torch.tensor(labels)


def load_images(filepaths):
    to_tensor = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    tensor = None

    for item in filepaths:
        try:
            image = Image.open(item).convert("RGB")
            img_tensor = to_tensor(image)

            if tensor is None:
                tensor = img_tensor.unsqueeze(0)
            else:
                tensor = torch.cat((tensor, img_tensor.unsqueeze(0)), dim=0)
        except Exception as e:
            print(f"Error loading image {item}: {e}")
    
    return tensor


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input: 3 x 32 x 32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate FC input size
        # 32x32 -> conv1 -> 32x32 -> pool -> 16x16
        # 16x16 -> conv2 -> 16x16 -> pool -> 8x8
        # 32 channels * 8 * 8 = 2048
        self.fc_input_size = 32 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4)
        
        self.fc1 = nn.Linear(in_features=self.fc_input_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, self.fc_input_size)

        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        
        return x


def train(model, criterion, optimizer, filepaths, labels, device):
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(NUM_EPOCHS):
        samples_trained = 0
        run_loss = 0
        correct_preds = 0
        total_samples = len(filepaths) 

        permutation = torch.randperm(total_samples)
        for i in range(0, total_samples, BATCH_SIZE):
            indices = permutation[i : i+BATCH_SIZE]
            batch_inputs = load_images(filepaths[indices])
            if batch_inputs is None: continue
            
            # Move data to device
            batch_inputs = batch_inputs.to(device)
            batch_labels = labels[indices].to(device)
            
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
            
            # Progress update
            if (i // BATCH_SIZE) % 10 == 0:
                print(f"  Epoch {epoch+1}: Batch {i // BATCH_SIZE} processed ({samples_trained}/{total_samples})")

        avg_loss = run_loss / (total_samples / BATCH_SIZE) 
        accuracy = correct_preds / float(samples_trained) 
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy.item())

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} " +
              f"({samples_trained}/{total_samples}): " +
              f"Loss={avg_loss:.5f}, Accuracy={accuracy:.5f}")
              
    return history


def test(model, filepaths, labels, device):
    samples_tested = 0
    correct_preds = 0
    total_samples = len(filepaths)
    
    all_preds = []
    all_labels = []
    incorrect_predictions = []

    for i in range(0, total_samples, BATCH_SIZE):
        batch_filepaths = filepaths[i : i + BATCH_SIZE]
        batch_inputs = load_images(batch_filepaths)
        if batch_inputs is None: continue
        
        # Move data to device
        batch_inputs = batch_inputs.to(device)
        batch_labels = labels[i : i + BATCH_SIZE].to(device)
        
        if len(batch_inputs) == 0: continue

        outputs = model(batch_inputs)

        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, dim=1)

        samples_tested += len(batch_labels)
        correct_preds += torch.sum(preds == batch_labels)
        
        all_preds.extend(preds.tolist())
        all_labels.extend(batch_labels.tolist())
        
        # Identify incorrect predictions
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
    
    if incorrect_predictions:
        print(f"\nIncorrect Predictions ({len(incorrect_predictions)}):")
        # Show first 10 to avoid clutter
        for item in incorrect_predictions[:10]:
            print(f"  {item['filename']}: True={item['true']}, Pred={item['pred']}")
        if len(incorrect_predictions) > 10:
            print(f"  ... and {len(incorrect_predictions) - 10} more.")
    else:
        print("\nNo incorrect predictions!")
        
    return all_labels, all_preds, accuracy


def plot_metrics(history, output_dir, output_dir_name, test_accuracy=None):
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS + 1), history['loss'], marker='o', label='Training Loss')
    plt.title(f'Training Loss\n({output_dir_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPOCHS + 1), history['accuracy'], marker='o', color='green', label='Training Accuracy')
    title = f'Training Accuracy\n({output_dir_name})'
    if test_accuracy is not None:
        title += f'\nTest Acc: {test_accuracy:.2%}'
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()
    print(f"Saved training metrics plot to {output_dir}/training_metrics.png")


def plot_confusion_matrix(y_true, y_pred, output_dir, output_dir_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=ID_TO_CLASS.values(), 
                yticklabels=ID_TO_CLASS.values())
    plt.title(f'Confusion Matrix\n({output_dir_name})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    print(f"Saved confusion matrix plot to {output_dir}/confusion_matrix.png")


def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Setup Directories
    # Create unique output directory based on time and global params
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_name = f"{TEAM_MEMBER_NAME}_{TRIAL_PARAM_NAME}_{timestamp}"
    output_dir = os.path.join("outputs", output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")

    # Define data paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Look for data_final
    dir_train = os.path.join(parent_dir, "data_final")
    if not os.path.exists(dir_train):
        dir_train = "data_final" # Fallback to current dir
        
    dir_test = os.path.join(parent_dir, "test")
    if not os.path.exists(dir_test):
        dir_test = "test"

    print(f"Training data path: {dir_train}")
    print(f"Test data path: {dir_test}")

    # 3. Prepare Data
    filepaths_train, labels_train = prepare_data(dir_train)
    filepaths_test, labels_test = prepare_data(dir_test)

    if len(filepaths_train) == 0:
        print("Error: No training images found!")
        return

    # 4. Initialize Model
    model = SimpleCNN()
    model.to(device) # Move model to device
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Train
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    history = train(model, criterion, optimizer, filepaths_train, labels_train, device)
    
    # Save Model
    model_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # 6. Test and Visualize
    test_acc = None
    if len(filepaths_test) > 0:
        print(f"Found {len(filepaths_test)} test images. Testing...")
        y_true, y_pred, test_acc = test(model, filepaths_test, labels_test, device)
        plot_confusion_matrix(y_true, y_pred, output_dir, output_dir_name)
    else:
        print("No test images found, skipping confusion matrix.")

    # 7. Plot Metrics (with Test Acc if available)
    plot_metrics(history, output_dir, output_dir_name, test_acc)

if __name__ == "__main__":
    main()
