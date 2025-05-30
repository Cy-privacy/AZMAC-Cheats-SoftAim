import os
import torch
import numpy as np
import logging

class YOLOHandler:
    def __init__(self, config_manager, models_path):
        self.config_manager = config_manager
        self.models_path = models_path
        self.model = None
        self.load_model()

    def load_model(self):
        self.model_existence_check()
        print(f"Loading {self.get_model_name()} with yolov5 for {self.config_manager.get_setting('yolo_mode')} inference.")

        try:
            # Try to load the model using PyTorch
            model_path = f"{self.models_path}/{self.get_model_name()}"
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=model_path,
                verbose=False,
                trust_repo=True,
                force_reload=True,
            )
            
            # Set confidence threshold
            self.model.conf = self.config_manager.get_setting("confidence") / 100
            self.model.iou = self.config_manager.get_setting("confidence") / 100
            
            print("Model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a simpler model loading approach
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print("Model loaded using ultralytics YOLO.")
            except Exception as e2:
                print(f"Failed to load model with alternative method: {e2}")
                self.model = None

    def detect(self, frame):
        if self.model is None:
            return np.zeros((0, 6))  # Return empty detections if model not loaded
            
        try:
            # Convert frame to RGB if it's in BGR format (OpenCV)
            if frame.shape[2] == 3:  # Check if it has 3 channels
                frame_rgb = frame  # Assume it's already in RGB format
                
            # Run inference
            results = self.model(frame_rgb, size=[self.config_manager.get_setting("height"), self.config_manager.get_setting("width")])
            
            # Extract detections
            return results.xyxy[0].cpu().numpy()  # Returns in format [x1, y1, x2, y2, confidence, class]
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return np.zeros((0, 6))  # Return empty detections on error

    def get_model_name(self):
        if self.config_manager.get_setting("yolo_mode") == "pytorch":
            return f"{self.config_manager.get_setting('yolo_model')}.pt"
        return f"{self.config_manager.get_setting('yolo_model')}.pt"

    def model_existence_check(self):
        model_path = f"{self.models_path}/{self.get_model_name()}"
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found. Please download the model and place it in the models directory.")
            # Create models directory if it doesn't exist
            os.makedirs(self.models_path, exist_ok=True)