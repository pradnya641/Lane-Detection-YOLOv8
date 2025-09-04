from ultralytics import YOLO
import os

# --- 1. DEFINE YOUR YAML CONTENT ---
# We use relative paths, which makes the project portable.
# The 'path' is the root directory of your dataset.
# The 'train' and 'val' paths are relative to the 'path'.
yaml_content = """
# This path points to our new split dataset folder
path: ./dataset_split
train: images/train
val: images/val

# Class info
nc: 1
names: ['lane']
"""

# The location where we will save the YAML file
yaml_path = os.path.join("dataset_split", "data.yaml")

# Write the YAML content to the file
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"✅ Successfully created {yaml_path}")


# --- 2. TRAIN THE MODEL ---
# This is the main entry point of the script
if __name__ == '__main__':
    # Load a model.
    # Choose one of the following:
    # 1. Start training from a pre-trained model
    model = YOLO('yolov8s-seg.pt')

    # 2. Resume training from your last checkpoint (if you have one)
    # model = YOLO('path/to/your/last.pt')

    print("Starting training...")
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=8, # Start with a smaller batch size like 8
        device=0, # Use '0' for GPU or 'cpu' for CPU
        workers=4, # A lower number like 4 is a safe start
        project='training_results',
        name='lane_detection_run'
    )

    print("✅ Training finished!")
    print("Results saved in the 'training_results' folder.")
