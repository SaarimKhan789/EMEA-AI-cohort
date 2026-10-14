from ultralytics import YOLO
import os
import glob

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train on COCO128
results = model.train(
    data="coco128.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    project="Assignment9",
    name="YOLOv8",
    exist_ok=True
)

# Path to trained model
best_pt = os.path.join(
    results.save_dir,
    "weights",
    "best.pt"
)

print("\nBest Model Path:")
print(best_pt)

# Load best model
best_model = YOLO(best_pt)

# Evaluate
metrics = best_model.val()

print("\n===== EVALUATION RESULTS =====")
print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")

# Find sample image
dataset_path = os.path.expanduser(
    "~/.config/Ultralytics/datasets/coco128/images/train2017/*.jpg"
)

images = glob.glob(dataset_path)

if images:

    test_img = images[0]

    prediction_results = best_model.predict(
        source=test_img,
        conf=0.25,
        save=True
    )

    print("\n===== DETECTIONS =====")

    for r in prediction_results:
        for box in r.boxes:

            cls = int(box.cls.item())
            conf = float(box.conf.item())

            print(
                f"{best_model.names[cls]} : {conf:.2f}"
            )

