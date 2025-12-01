import os
import random
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.v2 import GaussianNoise, functional

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "/Users/tohyue-sheng/git/FruitClassifier_CNN_SA41110ML_CA/data_original/train"  # folder containing all images
OUTPUT_DIR = "./augmented_images"
CLASSES = ["apple", "banana", "orange", "mixed"]
NUM_IMAGES_PER_CLASS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def get_random_images(data_dir, cls_name, n=10):
    files = [f for f in os.listdir(data_dir) if cls_name in f.lower()]
    return random.sample(files, min(n, len(files)))

# ----------------------------
# TRANSFORMS
# ----------------------------
def apply_transforms(pil_img):
    """Apply flip, color jitter, and gaussian noise without resizing"""
    # Random horizontal flip
    pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random color jitter
    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    pil_img = color_jitter(pil_img)

    # Apply GaussianNoise
    gaussian = GaussianNoise(mean=0.0, sigma=0.05, clip=True)
    img_tensor = gaussian(transforms.ToTensor()(pil_img))  # no need for functional.to_tensor

     # Random cutout (applied to all images, random position)
    random_cutout = transforms.RandomErasing(
        p=1.0,               # always applied
        scale=(0.02, 0.05),   # fraction of image area
        ratio=(0.3, 2),    # aspect ratio of rectangle
        value=0,             # fill with black
        inplace=True
    )
    img_tensor = random_cutout(img_tensor)

    # Convert back to PIL
    pil_out = transforms.ToPILImage()(img_tensor)

    return pil_out

# ----------------------------
# MAIN LOOP
# ----------------------------
for cls in CLASSES:
    img_paths = get_random_images(DATA_DIR, cls, NUM_IMAGES_PER_CLASS)
    cls_output_dir = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(cls_output_dir, exist_ok=True)

    for img_name in img_paths:
        img_path = os.path.join(DATA_DIR, img_name)
        pil_img = Image.open(img_path).convert("RGB")
        aug_img = apply_transforms(pil_img)

        output_path = os.path.join(cls_output_dir, f"aug_{img_name}")
        aug_img.save(output_path)

        print(f"Saved augmented image: {output_path}")

print("All images augmented and saved!")
