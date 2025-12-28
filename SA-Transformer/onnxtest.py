import onnxruntime as ort
import numpy as np
import yaml     
from preprocess import preprocess_onnx
import time

# Load Configuration from YAML
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load raw data
test = np.load('../data/SAAssignment2025/test.npy', allow_pickle=True)
# Use batch_size=1 for ONNX inference (model was exported with fixed batch size)
test_loader = preprocess_onnx(test, batch_size=1, scaler_save_path='scaler.pkl')

# Path to ONNX model file
onnx_model_path = 'SA-Transformer.onnx'

# Load ONNX model
onnx_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Print which provider is actually being used
print(f"Using provider: {onnx_session.get_providers()}")

# Get model input and output names
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

correct = 0
total = 0

# Warm-up
dummy_batch = next(iter(test_loader))[0].numpy().astype(np.float32)
onnx_session.run([output_name], {input_name: dummy_batch})

print("Starting timed inference...")
start_time = time.perf_counter()

for images, labels in test_loader:
    input_data = images.numpy().astype(np.float32)

    result = onnx_session.run([output_name], {input_name: input_data})[0]

    predicted = np.argmax(result, axis=1)
    total += labels.size(0)
    correct += (predicted == labels.numpy()).sum().item()

end_time = time.perf_counter()
total_time = end_time - start_time

print(f"\n--- Performance Report ---")
print(f"Total time for {len(test_loader)} batches: {total_time:.4f} seconds")
print(f"Average time per batch: {(total_time / len(test_loader)) * 1000:.2f} ms")
print(f"Throughput: {total / total_time:.2f} samples/sec")
print(f"ONNX Model Accuracy: {100 * correct / total:.2f}%")

