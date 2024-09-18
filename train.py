import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Constants
NUM_CLASSES = 102
BATCH_SIZE = 32
NUM_EPOCHS = 35
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_data_loaders(data_dir):
    """
    Create and return data loaders for training and validation.
    
    Args:
    data_dir (str): Path to the dataset directory
    
    Returns:
    tuple: train_loader, val_loader
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    
    return dataloaders['train'], dataloaders['val']

def create_model():
    """
    Create and return a modified ResNet50 model.
    
    Returns:
    nn.Module: Modified ResNet50 model
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, NUM_CLASSES)
    )
    return model.to(DEVICE)

def train_model(model, train_loader, val_loader, criterion, optimizer):
    """
    Train the model and perform validation.
    
    Args:
    model (nn.Module): The neural network model
    train_loader (DataLoader): Training data loader
    val_loader (DataLoader): Validation data loader
    criterion: Loss function
    optimizer: Optimization algorithm
    
    Returns:
    nn.Module: Trained model
    """
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Training loop
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Training loss: {epoch_loss:.4f}")
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
    
    return model

def main():
    """
    Main function to orchestrate the training process.
    """
    # Setting up data loaders
    data_dir = "flower_data"
    train_loader, val_loader = create_data_loaders(data_dir)
    
    # Creating model, loss function, and optimizer
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    # Training the model
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer)
    
    # Saving the trained model
    torch.save(trained_model.state_dict(), "models/resnet50_flowers.pth")
    print("Model saved")

if __name__ == "__main__":
    main()