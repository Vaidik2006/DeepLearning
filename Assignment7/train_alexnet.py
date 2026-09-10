import os
import time
import copy
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# 1. SETTINGS
# ============================================================

TRAIN_DIR = "Dataset/Split/train"
VAL_DIR = "Dataset/Split/val"
TEST_DIR = "Dataset/Split/test"

MODEL_PATH = "models/alexnet_grapevine_best.pth"
RESULTS_DIR = "results"

BATCH_SIZE = 32
EPOCHS = 15

CLASSIFIER_LR = 0.001
FEATURE_LR = 0.0001

SEED = 42


# ============================================================
# 2. CREATE DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 4. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("ALEXNET TRANSFER LEARNING")
print("=" * 60)

print("Device:", device)


# ============================================================
# 5. DATA TRANSFORMS
# ============================================================

# Training augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation and testing should NOT use random augmentation
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 6. LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=val_test_transform
)


# ============================================================
# 7. CHECK CLASSES
# ============================================================

class_names = train_dataset.classes

print("\nClasses:")
for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")


# ============================================================
# 8. DATASET SIZES
# ============================================================

print("\nDataset sizes:")
print("Training   :", len(train_dataset))
print("Validation :", len(val_dataset))
print("Testing    :", len(test_dataset))


# ============================================================
# 9. VERIFY CLASS ORDER
# ============================================================

if train_dataset.classes != val_dataset.classes:
    raise ValueError("Training and validation classes do not match!")

if train_dataset.classes != test_dataset.classes:
    raise ValueError("Training and testing classes do not match!")


# ============================================================
# 10. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 11. LOAD PRETRAINED ALEXNET
# ============================================================

print("\nLoading pretrained AlexNet...")

model = models.alexnet(
    weights=models.AlexNet_Weights.DEFAULT
)


# ============================================================
# 12. FREEZE FEATURES
# ============================================================

for param in model.features.parameters():
    param.requires_grad = False


# ============================================================
# 13. UNFREEZE DEEPER CONVOLUTIONAL LAYERS
# ============================================================

for param in model.features[8:].parameters():
    param.requires_grad = True


# ============================================================
# 14. CHANGE FINAL CLASSIFIER
# ============================================================

num_features = model.classifier[6].in_features

model.classifier[6] = nn.Linear(
    num_features,
    len(class_names)
)

model = model.to(device)


# ============================================================
# 15. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 16. OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    [
        {
            "params": model.features[8:].parameters(),
            "lr": FEATURE_LR
        },

        {
            "params": model.classifier.parameters(),
            "lr": CLASSIFIER_LR
        }
    ],

    weight_decay=0.0001
)


# ============================================================
# 17. LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# 18. TRAINING VARIABLES
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

best_val_accuracy = 0.0
best_epoch = 0

best_model_weights = copy.deepcopy(model.state_dict())


# ============================================================
# 19. TRAINING
# ============================================================

print("\nStarting training...")
print("=" * 60)

start_time = time.time()


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


    train_loss = running_loss / len(train_dataset)

    train_accuracy = 100 * correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()


    val_loss = running_loss / len(val_dataset)

    val_accuracy = 100 * correct / total


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)


    # --------------------------------------------------------
    # LEARNING RATE UPDATE
    # --------------------------------------------------------

    scheduler.step(val_accuracy)


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    best_marker = ""

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_epoch = epoch + 1

        best_model_weights = copy.deepcopy(model.state_dict())

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        best_marker = " ⭐ BEST"


    # --------------------------------------------------------
    # PRINT EPOCH RESULTS
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
        f"{best_marker}"
    )


# ============================================================
# 20. TRAINING TIME
# ============================================================

training_time = time.time() - start_time

print("\nTraining completed.")
print(f"Training Time: {training_time:.2f} seconds")


# ============================================================
# 21. LOAD BEST MODEL
# ============================================================

print("\nLoading best validation model...")

model.load_state_dict(best_model_weights)

model.eval()


# ============================================================
# 22. FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_correct = 0
test_total = 0

all_predictions = []
all_labels = []

test_loss_total = 0.0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        test_loss_total += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        test_total += labels.size(0)

        test_correct += (predicted == labels).sum().item()

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


test_loss = test_loss_total / len(test_dataset)

test_accuracy = 100 * test_correct / test_total


print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.2f}%")


# ============================================================
# 23. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=2
)

print("\nCLASSIFICATION REPORT")
print("=" * 60)
print(report)


# ============================================================
# 24. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(8, 6))

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45
)

plt.title("AlexNet - Test Set Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "alexnet_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 25. ACCURACY GRAPH
# ============================================================

epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(8, 6))

plt.plot(
    epochs_range,
    train_accuracies,
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title("AlexNet - Training and Validation Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "alexnet_accuracy.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 26. LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    epochs_range,
    train_losses,
    label="Training Loss"
)

plt.plot(
    epochs_range,
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("AlexNet - Training and Validation Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "alexnet_loss.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 27. SAVE RESULTS
# ============================================================

results_file = os.path.join(
    RESULTS_DIR,
    "alexnet_results.txt"
)

with open(results_file, "w") as f:

    f.write("ALEXNET TRANSFER LEARNING RESULTS\n")
    f.write("=" * 50 + "\n\n")

    f.write("Dataset: Grapevine Leaves\n")

    f.write(
        f"Number of Classes: {len(class_names)}\n"
    )

    f.write(
        f"Training Images: {len(train_dataset)}\n"
    )

    f.write(
        f"Validation Images: {len(val_dataset)}\n"
    )

    f.write(
        f"Testing Images: {len(test_dataset)}\n"
    )

    f.write(
        f"Epochs: {EPOCHS}\n"
    )

    f.write(
        f"Batch Size: {BATCH_SIZE}\n"
    )

    f.write(
        f"Best Epoch: {best_epoch}\n"
    )

    f.write(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy:.2f}%\n"
    )

    f.write(
        f"Final Test Accuracy: "
        f"{test_accuracy:.2f}%\n"
    )

    f.write(
        f"Test Loss: {test_loss:.4f}\n"
    )

    f.write(
        f"Training Time: "
        f"{training_time:.2f} seconds\n"
    )

    f.write("\n\nCLASSIFICATION REPORT\n")
    f.write("=" * 50 + "\n\n")

    f.write(report)


# ============================================================
# 28. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ALEXNET TRAINING AND TESTING COMPLETE")
print("=" * 60)

print(f"Best Validation Accuracy : {best_val_accuracy:.2f}%")
print(f"Best Epoch               : {best_epoch}")
print(f"Final Test Accuracy      : {test_accuracy:.2f}%")
print(f"Test Loss                : {test_loss:.4f}")
print(f"Training Time             : {training_time:.2f} seconds")

print("\nSaved model:")
print(MODEL_PATH)

print("\nSaved results:")
print(results_file)

print("\nSaved graphs:")
print("results/alexnet_accuracy.png")
print("results/alexnet_loss.png")
print("results/alexnet_confusion_matrix.png")

print("=" * 60)