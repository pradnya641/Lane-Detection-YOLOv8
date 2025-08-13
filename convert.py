import os
from pathlib import Path
import cv2
from tqdm import tqdm

# --- CONFIGURATION ---
CLEAN_DATA_DIR = r'C:\Users\puroh\LaneATT\culane_clean_data'
OUTPUT_DATASET_DIR = r'C:\Users\puroh\LaneATT\datasets\lane_dataset'
# ---------------------

def convert_clean_data_to_yolo(data_dir, output_dir):
    print(f"Reading all data from the clean folder: {data_dir}")
    images_out_path = Path(output_dir) / 'images' / 'train'
    labels_out_path = Path(output_dir) / 'labels' / 'train'
    images_out_path.mkdir(parents=True, exist_ok=True)
    labels_out_path.mkdir(parents=True, exist_ok=True)

    label_files = list(Path(data_dir).glob('*.lines.txt'))
    print(f"Found {len(label_files)} label files in the clean folder.")

    if not label_files:
        print("Error: No '.lines.txt' files found.")
        return

    processed_count = 0
    for label_file in tqdm(label_files, desc="Converting files"):
        
        # --- THIS IS THE CORRECTED CODE ---
        base_name = label_file.name.replace('.lines.txt', '')
        image_path = label_file.parent / f"{base_name}.jpg"
        # ------------------------------------

        if not image_path.exists():
            continue

        with open(label_file, 'r') as f:
            points_str = f.read().strip().split()
            if not points_str:
                continue

        all_points = []
        for i in range(0, len(points_str), 2):
            all_points.append(float(points_str[i]))
            all_points.append(float(points_str[i+1]))

        if len(all_points) < 4:
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue
        img_h, img_w, _ = img.shape
        
        output_label_file = labels_out_path / f"{base_name}.txt"
        with open(output_label_file, 'w') as out_f:
            class_id = 0
            normalized_line = [str(class_id)]
            for i in range(0, len(all_points), 2):
                x_norm = all_points[i] / img_w
                y_norm = all_points[i+1] / img_h
                normalized_line.append(f"{x_norm:.6f}")
                normalized_line.append(f"{y_norm:.6f}")
            out_f.write(" ".join(normalized_line))
        
        cv2.imwrite(str(images_out_path / f"{base_name}.jpg"), img)
        processed_count += 1

    print(f"\nConversion complete! Processed {processed_count} valid image/label pairs.")

if __name__ == '__main__':
    convert_clean_data_to_yolo(CLEAN_DATA_DIR, OUTPUT_DATASET_DIR)