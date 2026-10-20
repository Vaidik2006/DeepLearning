import os
import shutil
import random

# ==========================================
# SETTINGS
# ==========================================

SOURCE_DIR = "Dataset/Grapevine_Leaves_Image_Dataset"
OUTPUT_DIR = "Dataset/Split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

# ==========================================
# CHECK RATIOS
# ==========================================

assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0

random.seed(SEED)

# ==========================================
# CREATE OUTPUT DIRECTORIES
# ==========================================

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# ==========================================
# GET CLASSES
# ==========================================

classes = sorted([
    folder for folder in os.listdir(SOURCE_DIR)
    if os.path.isdir(os.path.join(SOURCE_DIR, folder))
])

print("Classes found:")
print(classes)

# ==========================================
# SPLIT EACH CLASS
# ==========================================

total_train = 0
total_val = 0
total_test = 0

for class_name in classes:

    source_class_dir = os.path.join(SOURCE_DIR, class_name)

    images = [
        file for file in os.listdir(source_class_dir)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    random.shuffle(images)

    total_images = len(images)

    train_end = int(total_images * TRAIN_RATIO)
    val_end = train_end + int(total_images * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Create class directories
    train_class_dir = os.path.join(OUTPUT_DIR, "train", class_name)
    val_class_dir = os.path.join(OUTPUT_DIR, "val", class_name)
    test_class_dir = os.path.join(OUTPUT_DIR, "test", class_name)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Copy files
    for image in train_images:
        shutil.copy2(
            os.path.join(source_class_dir, image),
            os.path.join(train_class_dir, image)
        )

    for image in val_images:
        shutil.copy2(
            os.path.join(source_class_dir, image),
            os.path.join(val_class_dir, image)
        )

    for image in test_images:
        shutil.copy2(
            os.path.join(source_class_dir, image),
            os.path.join(test_class_dir, image)
        )

    total_train += len(train_images)
    total_val += len(val_images)
    total_test += len(test_images)

    print(
        f"{class_name}: "
        f"Train={len(train_images)}, "
        f"Val={len(val_images)}, "
        f"Test={len(test_images)}"
    )

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 50)
print("DATASET SPLIT COMPLETE")
print("=" * 50)

print(f"Training images   : {total_train}")
print(f"Validation images : {total_val}")
print(f"Testing images    : {total_test}")
print(f"Total images      : {total_train + total_val + total_test}")

print("=" * 50)