from ultralytics import YOLO
import cv2

# --- CONFIGURATION ---
# Path to the 'best.pt' model you downloaded from Colab
MODEL_PATH = r'C:\Users\puroh\LaneATT\best.pt'

# Path to a NEW test image (one that was NOT in your dataset)
IMAGE_PATH = r'C:\Users\puroh\LaneATT\my_new_test_image.jpg' # <-- IMPORTANT: CHANGE THIS
# ---------------------

def main():
    print(f"Loading your trained model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    print(f"Running lane detection on {IMAGE_PATH}...")
    # Set save=True to save the image with the detected lanes drawn on it
    results = model.predict(source=IMAGE_PATH, save=True)
    
    print("\nPrediction complete!")
    # The result will be saved in a new 'runs/segment/predict' folder.
    if results and results[0].save_dir:
        print(f"Result image saved to: {results[0].save_dir}")
    else:
         print("Could not determine save directory.")

if __name__ == '__main__':
    main()