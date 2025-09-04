from ultralytics import YOLO

# --- 1. DEFINE YOUR PATHS ---
# Path to your BEST trained model weights
best_model_path = 'training_results/lane_detection_run5/weights/best.pt'

# Path to the video you want to test
source_video_path = 'delhiroad.mp4'


# --- 2. LOAD YOUR CUSTOM MODEL ---
print(f"Loading model from: {best_model_path}")
model = YOLO(best_model_path)


# --- 3. RUN PREDICTION ---
print(f"Running prediction on: {source_video_path}")
results = model.predict(
    source=source_video_path,
    save=True  # This saves the output video with detections
)

print("\n✅ Prediction complete!")
print("Your output video has been saved in the 'runs/detect/predict' folder.")
