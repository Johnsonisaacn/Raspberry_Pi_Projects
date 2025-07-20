import time
import torch
import numpy as np
from torchvision import models, transforms
import cv2
from PIL import Image

# Load ImageNet class labels
LABELS_URL = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
import requests

# Download labels (run once)
response = requests.get(LABELS_URL)
labels = eval(response.text)  # Convert string to dictionary

# Initialize camera and model
torch.backends.quantized.engine = 'qnnpack'
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
net = torch.jit.script(net)  # Optimize with JIT

# Inference loop
with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame")

        # Preprocess frame (BGR → RGB → Tensor)
        rgb_frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
        input_tensor = preprocess(rgb_frame)
        input_batch = input_tensor.unsqueeze(0)

        # Run inference
        output = net(input_batch)
        _, predicted_idx = torch.max(output, 1)
        predicted_label = labels[predicted_idx.item()]

        # Overlay label on frame
        cv2.putText(
            frame, 
            f"Predicted: {predicted_label}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0),  # Green text
            2
        )

        # Display frame
        cv2.imshow("Raspberry Pi Object Detection", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()