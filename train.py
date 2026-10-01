"""
Model Training Script for Skin Disease Classification
-----------------------------------------------------
- Backbone: MobileNetV2 with Transfer Learning
- Classes: Acne vs. Other Skin Lesion (Balanced 50/50)
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Data Augmentation: Random Flips, Rotations, Color Jitter
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

DATA_DIR = "data"
MODEL_SAVE_PATH = "skin_model.pth"
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=0)
        for x in ['train', 'val']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names

def train_model():
    print(f"[INFO] Training on device: {DEVICE}")
    
    if not os.path.exists(DATA_DIR) or not os.path.exists(os.path.join(DATA_DIR, 'train')):
        print(f"[ERROR] Data folder '{DATA_DIR}/train' not found.")
        print("Please run 'python prepare_data.py' first to balance and split your raw images.")
        return

    dataloaders, dataset_sizes, class_names = get_data_loaders()
    print(f"[INFO] Target Classes: {class_names}")
    print(f"[INFO] Train samples: {dataset_sizes['train']}, Val samples: {dataset_sizes['val']}")

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze base feature layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head for 2 classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n[SUCCESS] Best Validation Accuracy: {best_acc:.4f}")
    torch.save(best_model_wts, MODEL_SAVE_PATH)
    print(f"[INFO] Best model weights saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
