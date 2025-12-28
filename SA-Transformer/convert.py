import torch
from model import TransformerModel
import numpy as np
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load data to determine input shape
train = np.load('../data/SAAssignment2025/train.npy', allow_pickle=True)

# Load actual class names from the saved file
class_names = np.load('../data/SAAssignment2025/class_names.npy', allow_pickle=True)
num_classes = len(class_names)
num_features = train.shape[1] - 1

print(f"Finished loading data. Features: {num_features}, Classes: {num_classes}")

model_path = 'SA-Transformer.pth'
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

model.load_state_dict(torch.load(model_path, map_location=device))

print("\nExporting Transformer model to ONNX")
onnx_path = model_path.replace(".pth", ".onnx")

# Try with fixed batch size first to avoid dynamic shape issues
dummy_input = torch.randn(size=(1, num_features), device=device, dtype=torch.float32)

model.eval()
try:
    # Export without dynamic axes first (fixed batch size)
    torch.onnx.export(
        model.to(device),
        dummy_input, 
        onnx_path, 
        verbose=False,
        input_names=['input'], 
        output_names=['output'],
        # Remove dynamic_axes for now to avoid reshape issues
        opset_version=17,
        do_constant_folding=True,
        export_params=True
    )
    print(f"Transformer model exported to ONNX at: {onnx_path}")
    print("Note: Model exported with fixed batch size. For variable batch sizes, retrain with batch_first=True in Transformer layers.")
except Exception as e:
    print(f"Error exporting model to ONNX: {e}")
    import traceback
    traceback.print_exc()

