import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from src.model import SimpleNN, ConvNet


def get_data_loaders(batch_size=64):
    """Load and preprocess MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, num_epochs=5, learning_rate=0.001):
    """Train the neural network"""
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Device configuration (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Training on: {device}")
    
    # Training history for plotting
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Use tqdm for progress bar
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (data, target) in enumerate(train_bar):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            # Update progress bar
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct_train/total_train:.2f}%'
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        test_accuracy = test_model(model, test_loader, device)
        
        # Store history
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')
    
    return train_losses, train_accuracies, test_accuracies


def test_model(model, test_loader, device):
    """Test the model and return accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def plot_training_history(train_losses, train_accuracies, test_accuracies):
    """Plot training history"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def save_model(model, filepath):
    """Save the trained model"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    num_epochs = 5
    learning_rate = 0.001
    model_type = "simple"  # "simple" or "conv"
    
    print("🚀 Starting PyTorch MNIST Training")
    print("=" * 50)
    
    # Load data
    print("📊 Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(batch_size)
    print(f"✅ Dataset loaded - Training samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    
    # Create model
    if model_type == "simple":
        model = SimpleNN()
    else:
        model = ConvNet()
    
    print(f"🧠 Created model: {model.__class__.__name__}")
    print(f"📝 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n🏋️ Starting training...")
    train_losses, train_accuracies, test_accuracies = train_model(
        model, train_loader, test_loader, num_epochs, learning_rate
    )
    
    # Final evaluation
    print("\n📊 Final Results:")
    print(f"✅ Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"✅ Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    
    # Save model
    model_filename = f'models/saved_models/mnist_{model_type}_model.pth'
    save_model(model, model_filename)
    
    # Plot training history
    print("📈 Generating training plots...")
    plot_training_history(train_losses, train_accuracies, test_accuracies)
    
    print("\n🎉 Training completed successfully!")
    print(f"💾 Model saved as: {model_filename}")
    print("📊 Training plots saved as: training_history.png")
