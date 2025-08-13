from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8 segmentation model
    model = YOLO('yolov8n-seg.pt')

    # Path to your dataset configuration file
    data_yaml_path = r'C:\Users\puroh\LaneATT\datasets\lane_dataset\data.yaml'

    print("Starting lane detection model training...")
    results = model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=640,
        batch=4,
        project='runs/segment',
        name='lane_final_training'
    )
    print("Training finished.")

if __name__ == '__main__':
    main()