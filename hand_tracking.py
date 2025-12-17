# 

# ===============================
# Imports
# ===============================
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import mediapipe as mp

# ===============================
# Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# ===============================
# Load class names
# ===============================
with open("class_names.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f]

num_classes = len(class_names)
print("Loaded classes:", class_names)

# ===============================
# Load trained model
# ===============================
base_model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
)

# Freeze backbone
for param in base_model.features.parameters():
    param.requires_grad = False

# Custom classifier (same as training)
base_model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(base_model.last_channel, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.4),
    nn.Linear(512, num_classes)
)

model = base_model.to(device)
model.load_state_dict(
    torch.load(r"C:\Users\menna\OneDrive\Desktop\ArSL---Arabic-Sign-Language-main\best_model.pth", map_location=device)
)
model.eval()

print("Model loaded successfully.")

# ===============================
# Preprocessing (MATCHES TRAINING)
# ===============================
img_size = 224

preprocess = transforms.Compose([
    transforms.Resize(int(img_size * 1.15)),   # 257
    transforms.CenterCrop(img_size),            # 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# MediaPipe Hands
# ===============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,        # faster
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===============================
# Webcam
# ===============================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows-safe

print("Press 'q' to quit.")

# ===============================
# Main loop
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape

            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            margin = 20
            x_min = max(x_min - margin, 0)
            y_min = max(y_min - margin, 0)
            x_max = min(x_max + margin, w)
            y_max = min(y_max + margin, h)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            hand_img = Image.fromarray(hand_img)
            hand_img = preprocess(hand_img).unsqueeze(0).to(device)

            # Inference
            with torch.amp.autocast(device_type=device.type):
                outputs = model(hand_img)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)

            label = class_names[pred.item()]
            confidence = conf.item()

            # Draw
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                          (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

    cv2.imshow("Arabic Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# Cleanup
# ===============================
cap.release()
cv2.destroyAllWindows()
