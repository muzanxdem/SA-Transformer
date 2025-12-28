import torch
import numpy as np
from preprocess import preprocess
import torch.optim as optim
from model import TransformerModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchsummary import summary
import wandb
import yaml

wandb.login(key="f3c52d906dd19b797b189d5640be154a572c4ece")

def train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epochs):
    """
    Trains the Transformer model for a given number of epochs.
    
    Args:
        model: The model to train.
        train_loader: The training data loader.
        val_loader: The validation data loader.
        device: The device to train on.
        criterion: The loss function.
        optimizer: The optimizer.
        scheduler: The scheduler.
        num_epochs: The number of epochs to train for.

    Returns:
        The trained model.
    """
    best_loss = float('inf')
    patience = 5
    no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            data, target = data.to(device), target.to(device).long()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, _, _ = evaluate_model(model, val_loader, device, criterion)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'SA-Transformer.pth')
            print(f"Model saved to SA-Transformer.pth")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Epoch {epoch+1}/{num_epochs}, best_val={best_loss:.6f}")
                break

        scheduler.step(val_loss)
        wandb.log({
            "Train Loss": train_loss, 
            "Val Loss": val_loss,
            "Learning Rate": optimizer.param_groups[0]['lr']
        })
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    model.load_state_dict(torch.load('SA-Transformer.pth', map_location=device))
    return model

def evaluate_model(model, dataloader, device, criterion=None):
    """
    Evaluates the model on a given dataloader.
    Can compute loss and/or return predictions and labels.

    Args:
        model: The model to evaluate.
        dataloader: The data loader to evaluate on.
        device: The device to evaluate on.
        criterion: The loss function.

    Returns:
        The average loss, all predictions, and all labels.
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for samples, labels in dataloader:
            samples = samples.to(device).float()
            labels = labels.to(device)

            predictions = model(samples)

            if criterion:
                loss = criterion(predictions, labels.long())
                total_loss += loss.item()

            preds = torch.argmax(predictions, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if criterion and len(dataloader) > 0 else 0
    return avg_loss, all_preds, all_labels

def test_and_report(model, test_loader, device, class_names):
    """
    Evaluates a model on the test set and prints a classification report
    and confusion matrix.

    Args:
        model: The model to evaluate.
        test_loader: The test data loader.
        device: The device to evaluate on.
        class_names: The class names.

    Returns:
        The accuracy of the model on the test set.
    """
    print("\n--- Starting Final Test ---")
    model.eval()

    all_preds, all_labels = [], []
    with torch.inference_mode():
        for samples, labels in test_loader:
            samples = samples.to(device)
            labels = labels.to(device).long()
            
            predictions = model(samples)
            
            preds = torch.argmax(predictions, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Final Test Accuracy: {acc*100:.2f}%")

    wandb.log({"Accuracy": acc*100})

    print('--- Classification Report ---')
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    print('--- Confusion Matrix ---')
    print(confusion_matrix(all_labels, all_preds))
    return acc


if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load data from data directory
    train = np.load('../data/SAAssignment2025/train.npy', allow_pickle=True)
    test = np.load('../data/SAAssignment2025/test.npy', allow_pickle=True)
    val = np.load('../data/SAAssignment2025/val.npy', allow_pickle=True)
    
    # Load actual class names from the saved file
    class_names = np.load('../data/SAAssignment2025/class_names.npy', allow_pickle=True)
    num_classes = len(class_names)
    print(f"Finished loading data. Found {num_classes} classes in the dataset.")
    
    # Verify label range
    train_labels = train[:, -1].astype(np.int64)
    test_labels = test[:, -1].astype(np.int64)
    val_labels = val[:, -1].astype(np.int64)
    max_label = max(train_labels.max(), test_labels.max(), val_labels.max())
    min_label = min(train_labels.min(), test_labels.min(), val_labels.min())
    print(f"Label range: {min_label} to {max_label} (expected: 0 to {num_classes-1})")
    
    if max_label >= num_classes or min_label < 0:
        raise ValueError(f"Labels out of range! Labels are in [{min_label}, {max_label}], but model expects [0, {num_classes-1}]")

    # Preprocess data
    train_loader, test_loader, val_loader = preprocess(
        train, test, val, 
        config['batch_size'], 
        scaler_save_path='scaler.pkl'
    )
    
    # Create Transformer model
    num_features = train.shape[1] - 1
    model = TransformerModel(
        input_features=num_features,
        num_classes=num_classes,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    num_epochs = config['num_epochs']
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Print model summary
    print("\n--- Model Architecture ---")
    print(f"Input features: {num_features}")
    print(f"Number of classes: {num_classes}")
    print(f"Model dimension (d_model): {config['d_model']}")
    print(f"Attention heads: {config['nhead']}")
    print(f"Encoder layers: {config['num_layers']}")
    print(f"Feedforward dimension: {config['dim_feedforward']}")
    print(f"Dropout: {config['dropout']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize WandB
    wandb.init(
        project="Obi_SA-Transformer",
        config={
            "batch_size": config['batch_size'],
            "num_epochs": config['num_epochs'],
            "learning_rate": config['learning_rate'],
            "d_model": config['d_model'],
            "nhead": config['nhead'],
            "num_layers": config['num_layers'],
            "dim_feedforward": config['dim_feedforward'],
            "dropout": config['dropout'],
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "architecture": "Transformer"
        }
    )

    # Train model
    wandb.watch(model, log="all")
    model = train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epochs)

    # Test model
    test_and_report(model, test_loader, device, class_names)

