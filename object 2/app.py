from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import os
import torch
from ultralytics import YOLO
from flask_cors import CORS  # ✅ Added CORS handling

app = Flask(__name__)

# ✅ Enable CORS to allow frontend requests
CORS(app, resources={r"/upload": {"origins": "*"}})

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# ✅ Load YOLO model
model = YOLO("yolov8s.pt")  # Use "yolov8m.pt" for even better accuracy

def detect_objects_yolo(image_path, output_path):
    try:
        # ✅ Run YOLO with optimized parameters
        results = model(image_path, imgsz=1280, conf=0.2, iou=0.4)

        image = cv2.imread(image_path)
        object_count = 0

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
            object_count = len(boxes)

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])  # Bounding box coordinates
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ✅ Display object count on the image
        cv2.putText(image, f"Objects: {object_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(output_path, image)

        return object_count, output_path
    except Exception as e:
        print("Error:", e)
        return -1, None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    output_path = os.path.join(PROCESSED_FOLDER, file.filename)
    count, processed_path = detect_objects_yolo(filepath, output_path)
    
    if count == -1:
        return jsonify({"error": "Processing failed"}), 500
    
    # ✅ Return full URL to processed image
    return jsonify({"image_url": f"http://localhost:5001/static/processed/{file.filename}", "object_count": count})

# ✅ Ensure static files are served properly
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
