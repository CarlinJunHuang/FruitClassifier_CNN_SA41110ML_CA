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
import time
import random

# ==========================================
# 1. Global Hyperparameters & Settings
#    (Modify these to change experiment)
# ==========================================

# --- Experiment Info ---
TEAM_MEMBER_NAME = "Your Name" 
TRIAL_DESCRIPTION = "Trial Description" 

# --- Training Config ---
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# --- Learning Rate Scheduler Config ---
# 0: None
# 1: StepLR (Decay every N epochs)
# 2: ExponentialLR (Decay every epoch)
# 3: CosineAnnealingLR (Cosine curve)
# 4: CosineAnnealingWarmRestarts (Cosine with restarts)
LR_SCHEDULER_TYPE = 0

# Params for StepLR - only when LR_SCHEDULER_TYPE = 1
LR_STEP_SIZE = 7       # Every N epochs
LR_GAMMA = 0.1         # Multiply LR by gamma

# Params for ExpLR - only when LR_SCHEDULER_TYPE = 2
LR_EXP_GAMMA = 0.95    # Decay factor per epoch

# Params for CosineAnnealingLR - only when LR_SCHEDULER_TYPE = 3
LR_T_MAX = 10          # T_max (usually equal to NUM_EPOCHS)

# Params for CosineAnnealingWarmRestarts - only when LR_SCHEDULER_TYPE = 4
LR_T_0 = 5             # First cycle step size
LR_T_MULT = 2          # Cycle expansion factor

WEIGHT_DECAY = 0    # L2 Regularization (0.0 to disable)
RANDOM_SEED = 42

# --- Dataset Config ---
IMAGE_SIZE = 64
VALIDATION_SPLIT = 0.2  # Fraction of training data to use for validation
DATA_SOURCE_DIR = "data_final" # Options: "data_final", "data_final_HJ"

# --- Network Architecture Config ---
# Convolutional Layers: [out_channels, out_channels, ...]
CONV_ARCH = [16, 32] 
KERNEL_SIZE = 3

# Fully Connected Layers: [neurons, neurons, ...] (Hidden layers only)
FC_ARCH = [128]

POOLING_TYPE = "Max"    # Options: "Max", "Avg"
DROPOUT_RATE = 0.0      # Dropout probability (0.0 means no dropout)


# Class mapping (Fixed)
ID_TO_CLASS = {0: "apple", 1: "banana", 2: "orange", 3: "mixed"}

# ==========================================
# 2. Helpers & Setup
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_label_from_filename(filename):
    class_map = {"apple": 0, "banana": 1, "orange": 2, "mixed": 3}
    try:
        class_name = filename.split("_")[0].lower()
        return class_map.get(class_name, -1)
    except:
        return -1

def split_dataset(images, labels, filenames, val_split):
    """Splits data into training and validation sets."""
    total_count = len(images)
    indices = list(range(total_count))
    random.shuffle(indices)
    
    val_count = int(total_count * val_split)
    train_count = total_count - val_count
    
    train_idx = indices[:train_count]
    val_idx = indices[train_count:]
    
    return (
        images[train_idx], labels[train_idx], [filenames[i] for i in train_idx],
        images[val_idx], labels[val_idx], [filenames[i] for i in val_idx]
    )

def load_images_from_folder(target_dir):
    """
    Loads all images from the folder into a single Tensor.
    Returns: (images_tensor, labels_tensor, filenames_list)
    """
    filepaths = []
    labels = []
    filenames = []

    if os.path.exists(target_dir):
        files = os.listdir(target_dir)
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            label = get_label_from_filename(file)
            if label != -1:
                filepaths.append(os.path.join(target_dir, file))
                labels.append(label)
                filenames.append(file)
    else:
        print(f"Warning: Directory {target_dir} not found.")
        return None, None, None

    if not filepaths:
        return None, None, None

    # Transform
    to_tensor = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    images_list = []
    valid_labels = []
    valid_filenames = []

    print(f"Pre-loading {len(filepaths)} images from {target_dir}...")
    
    for i, item in enumerate(filepaths):
        try:
            image = Image.open(item).convert("RGB")
            img_tensor = to_tensor(image)
            images_list.append(img_tensor)
            valid_labels.append(labels[i])
            valid_filenames.append(filenames[i])
        except Exception as e:
            print(f"Error loading image {item}: {e}")

    if not images_list:
        return None, None, None

    # Stack into a single tensor (N, 3, H, W)
    images_tensor = torch.stack(images_list)
    labels_tensor = torch.tensor(valid_labels)
    
    return images_tensor, labels_tensor, valid_filenames

# ==========================================
# 3. Model Definition (Dynamic)
# ==========================================

class FlexibleCNN(nn.Module):
    def __init__(self):
        super(FlexibleCNN, self).__init__()
        
        # --- 1. Convolutional Block Construction ---
        self.conv_layers = nn.ModuleList()
        in_channels = 3 # RGB
        
        for out_channels in CONV_ARCH:
            # Conv + ReLU + Pool + Dropout(Optional)
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL_SIZE, padding=1)
            )
            in_channels = out_channels # Next layer input = current output
        
        # Store components to reuse
        self.relu = nn.ReLU()
        if POOLING_TYPE.lower() == "max":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif POOLING_TYPE.lower() == "avg":
             self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(f"Unknown POOLING_TYPE: {POOLING_TYPE}")
            
        self.dropout = nn.Dropout(p=DROPOUT_RATE) if DROPOUT_RATE > 0 else nn.Identity()
        
        # --- 2. Calculate Flatten Size ---
        self.fc_input_size = self._calculate_fc_input_size()
        
        # --- 3. Fully Connected Block Construction ---
        self.fc_layers = nn.ModuleList()
        in_features = self.fc_input_size
        
        for hidden_neurons in FC_ARCH:
            self.fc_layers.append(nn.Linear(in_features, hidden_neurons))
            in_features = hidden_neurons
            
        # Final Output Layer (Class Count)
        self.final_fc = nn.Linear(in_features, 4)

    def _calculate_output_size(self, input_size, kernel_size, stride=1, padding=0):
        return (input_size + 2 * padding - kernel_size) // stride + 1
    
    def _calculate_fc_input_size(self):
        """Simulate forward pass dimensions to determine FC input size"""
        size = IMAGE_SIZE
        # For each conv layer, we apply Conv -> Pool
        for _ in CONV_ARCH:
            # Conv (3x3, pad=1) -> Size change? (Assuming pad=1 keeps size SAME for k=3, s=1)
            size = self._calculate_output_size(size, KERNEL_SIZE, stride=1, padding=1)
            # Pool (2x2, stride=2)
            size = self._calculate_output_size(size, kernel_size=2, stride=2, padding=0)
            
        final_channels = CONV_ARCH[-1]
        return final_channels * size * size

    def forward(self, x):
        # 1. Conv Loop
        for conv in self.conv_layers:
            x = conv(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.dropout(x)

        # 2. Flatten
        x = x.view(-1, self.fc_input_size)

        # 3. FC Loop (Hidden)
        for fc in self.fc_layers:
            x = fc(x)
            x = self.relu(x)
            x = self.dropout(x)
            
        # 4. Final Output
        x = self.final_fc(x)
        return x

# ==========================================
# 4. Training & Testing Loops
# ==========================================

def train_epoch(model, criterion, optimizer, images, labels):
    """Runs one epoch of training"""
    model.train()
    total_loss = 0
    correct_preds = 0
    total_samples = len(images)
    
    # Shuffle
    permutation = torch.randperm(total_samples)
    
    for i in range(0, total_samples, BATCH_SIZE):
        indices = permutation[i : i+BATCH_SIZE]
        batch_inputs = images[indices]
        batch_labels = labels[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch_labels) # Weighted sum
        
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == batch_labels).item()
        
    avg_loss = total_loss / total_samples
    accuracy = correct_preds / total_samples
    return avg_loss, accuracy

def validate_epoch(model, criterion, images, labels):
    """Runs one epoch of validation (no grad)"""
    model.eval()
    total_loss = 0
    correct_preds = 0
    total_samples = len(images)
    
    with torch.no_grad():
        for i in range(0, total_samples, BATCH_SIZE):
            batch_inputs = images[i : i+BATCH_SIZE]
            batch_labels = labels[i : i+BATCH_SIZE]
            
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            
            total_loss += loss.item() * len(batch_labels)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == batch_labels).item()
            
    avg_loss = total_loss / total_samples
    accuracy = correct_preds / total_samples
    return avg_loss, accuracy

def run_training(model, device, images_train, labels_train, images_val=None, labels_val=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Select Scheduler
    scheduler = None
    if LR_SCHEDULER_TYPE == 1:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
        print(f"Using StepLR (step={LR_STEP_SIZE}, gamma={LR_GAMMA})")
    elif LR_SCHEDULER_TYPE == 2:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_EXP_GAMMA)
        print(f"Using ExponentialLR (gamma={LR_EXP_GAMMA})")
    elif LR_SCHEDULER_TYPE == 3:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LR_T_MAX)
        print(f"Using CosineAnnealingLR (T_max={LR_T_MAX})")
    elif LR_SCHEDULER_TYPE == 4:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=LR_T_0, T_mult=LR_T_MULT)
        print(f"Using CosineAnnealingWarmRestarts (T_0={LR_T_0}, T_mult={LR_T_MULT})")
    else:
        print("Using Constant Learning Rate (No Scheduler)")
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        # Train
        t_loss, t_acc = train_epoch(model, criterion, optimizer, images_train, labels_train)
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        
        # Validate
        val_log = ""
        if images_val is not None:
            v_loss, v_acc = validate_epoch(model, criterion, images_val, labels_val)
            history['val_loss'].append(v_loss)
            history['val_acc'].append(v_acc)
            val_log = f" | Val Loss={v_loss:.4f}, Val Acc={v_acc:.4f}"
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss={t_loss:.4f}, Train Acc={t_acc:.4f}{val_log}")
        
        # Step LR Scheduler
        if scheduler:
            scheduler.step()

    duration = time.time() - start_time
    print(f"\nTraining completed in {duration:.1f} seconds.")
    return history

def test_model(model, images, labels, filenames):
    model.eval()
    all_preds = []
    all_labels = []
    incorrect_predictions = []
    
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, dim=1)
        
        acc = torch.sum(preds == labels).item() / len(labels)
        
        all_preds = preds.cpu().numpy()
        all_labels = labels.cpu().numpy()
        
        # Find incorrect
        wrong_idx = (preds != labels).nonzero(as_tuple=True)[0]
        for idx in wrong_idx:
            i = idx.item()
            incorrect_predictions.append({
                "filename": filenames[i],
                "true": ID_TO_CLASS[labels[i].item()],
                "pred": ID_TO_CLASS[preds[i].item()]
            })

    print(f"Test Accuracy: {acc:.2%}")
    if incorrect_predictions:
        print(f"Incorrect predictions: {len(incorrect_predictions)}")
        for item in incorrect_predictions:
            print(f"  {item['filename']}: True={item['true']}, Pred={item['pred']}")
        
    return all_labels, all_preds, acc

# ==========================================
# 5. Visualization (With Parameters)
# ==========================================

def get_param_summary_text():
    """Generates a text summary of global parameters."""
    
    sched_info = "None"
    if LR_SCHEDULER_TYPE == 1: sched_info = f"StepLR(sz={LR_STEP_SIZE},g={LR_GAMMA})"
    elif LR_SCHEDULER_TYPE == 2: sched_info = f"ExpLR(g={LR_EXP_GAMMA})"
    elif LR_SCHEDULER_TYPE == 3: sched_info = f"Cos(Tmax={LR_T_MAX})"
    elif LR_SCHEDULER_TYPE == 4: sched_info = f"CosWarm(T0={LR_T_0},Tm={LR_T_MULT})"

    text = (
        f"Epochs: {NUM_EPOCHS}\n"
        f"Batch: {BATCH_SIZE}\n"
        f"LR: {LEARNING_RATE}\n"
        f"Sched: {sched_info}\n"
        f"W-Decay: {WEIGHT_DECAY}\n"
        f"ImgSize: {IMAGE_SIZE}\n"
        f"ConvArch: {CONV_ARCH}\n"
        f"FCArch: {FC_ARCH}\n"
        f"Pool: {POOLING_TYPE}\n"
        f"Drop: {DROPOUT_RATE}"
    )
    return text

def plot_metrics(history, output_dir, output_dir_name, test_acc=None):
    plt.figure(figsize=(14, 6))
    epochs = range(1, NUM_EPOCHS + 1)
    
    # 1. Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    if history['val_loss']:
        plt.plot(epochs, history['val_loss'], 'r--s', label='Val Loss')
    plt.title(f'Loss Curve\n{output_dir_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'g-o', label='Train Acc')
    if history['val_acc']:
        plt.plot(epochs, history['val_acc'], 'm--s', label='Val Acc')
        
    title_str = f'Accuracy Curve'
    if test_acc:
        title_str += f"\nTest Result: {test_acc:.2%}"
        
    plt.title(title_str)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. Add Parameter Text Box
    # Place it outside the plot to the right
    param_text = get_param_summary_text()
    
    # Adjusted position to be safer (0.8 instead of 0.92) and smaller font for tighter spacing
    plt.figtext(0.85, 0.5, param_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8),
                va="center")
    
    # Adjust right margin to make space for the text box
    plt.subplots_adjust(right=0.8) 
    
    save_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Metrics plot saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, output_dir, output_dir_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=ID_TO_CLASS.values(), 
                yticklabels=ID_TO_CLASS.values())
    plt.title(f'Confusion Matrix\n{output_dir_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

# ==========================================
# 6. Main Execution
# ==========================================

def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup Output Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_name = f"{TEAM_MEMBER_NAME}_{TRIAL_DESCRIPTION}_{timestamp}"
    output_dir = os.path.join("outputs", output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output folder: {output_dir}")

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    dir_train_full = os.path.join(parent_dir, DATA_SOURCE_DIR)
    dir_test = os.path.join(parent_dir, "test")
    
    if not os.path.exists(dir_train_full):
        # Fallback for relative paths if script is run differently
        if os.path.exists(DATA_SOURCE_DIR): dir_train_full = DATA_SOURCE_DIR
        else:
            print(f"Error: Training directory '{DATA_SOURCE_DIR}' not found.")
            return

    # Load All Training Data
    print("\nLoading dataset...")
    images, labels, filenames = load_images_from_folder(dir_train_full)
    if images is None: return
    
    images = images.to(device)
    labels = labels.to(device)
    
    # Validate Split
    if VALIDATION_SPLIT > 0:
        print(f"Splitting data: {1.0-VALIDATION_SPLIT:.0%} Train, {VALIDATION_SPLIT:.0%} Val")
        img_train, lbl_train, _, img_val, lbl_val, _ = split_dataset(images, labels, filenames, VALIDATION_SPLIT)
    else:
        print("Validation Split is 0. Using ALL data for Training.")
        img_train, lbl_train = images, labels
        img_val, lbl_val = None, None
        
    print(f"Train set: {len(img_train)} samples")
    if img_val is not None: print(f"Val set:   {len(img_val)} samples")
    
    # Initialize Model
    model = FlexibleCNN().to(device)
    
    # Train
    history = run_training(model, device, img_train, lbl_train, img_val, lbl_val)
    
    # Save Model
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    
    # Test
    test_acc = None
    if os.path.exists(dir_test):
        print("\nLoading test set...")
        t_imgs, t_lbls, t_files = load_images_from_folder(dir_test)
        if t_imgs is not None:
             t_imgs, t_lbls = t_imgs.to(device), t_lbls.to(device)
             print(f"Test set:  {len(t_imgs)} samples")
             y_true, y_pred, test_acc = test_model(model, t_imgs, t_lbls, t_files)
             plot_confusion_matrix(y_true, y_pred, output_dir, output_dir_name)
    else:
        print("Test folder not found, skipping final test.")

    # Plot
    plot_metrics(history, output_dir, output_dir_name, test_acc)
    print("\nDone!")

if __name__ == "__main__":
    main()
