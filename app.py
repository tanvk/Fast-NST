# app.py
import os
import asyncio
import uvicorn
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from pydantic import BaseModel
import json
import torch
import utils
import transformer
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time
import re  # For handling data URLs

# Create output directories
os.makedirs("static", exist_ok=True)

app = FastAPI(title="Neural Style Transfer App")

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Available styles
STYLES = {
    "starry": "transforms/starry.pth",
    "mosaic": "transforms/mosaic.pth",
    "wave": "transforms/wave.pth",
    "tokyo_ghoul": "transforms/tokyo_ghoul.pth",
    "udnie": "transforms/udnie.pth",
    "lazy": "transforms/lazy.pth"
}

TEMPORAL_WEIGHT = 0.5  # Weight for temporal consistency

# Classes from COCO dataset
AVAILABLE_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
]

class StyleOptions(BaseModel):
    mode: str  # 'static' or 'semantic'
    fg_style: Optional[str] = None  # style for foreground (only for semantic mode)
    bg_style: str  # style for background
    target_classes: List[str] = ["person"]  # objects to preserve (only for semantic mode)
    preserve_color_fg: bool = False
    preserve_color_bg: bool = False

# Object detector class
class ObjectDetector:
    def __init__(self, device=None):
        """Initialize object detector model"""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for object detection: {self.device}")
        
        # Load pre-trained model
        print("Loading object detection model...")
        self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define transformation
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # COCO class names
        self.class_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def detect_objects(self, frame):
        """
        Detect objects in a frame
        
        Args:
            frame: OpenCV BGR frame
            
        Returns:
            List of detected objects with boxes, masks, and class info
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        img_tensor = self.transform(rgb_frame).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            prediction = self.model([img_tensor])
        
        # Process results
        objects = []
        for i in range(len(prediction[0]['boxes'])):
            score = prediction[0]['scores'][i].cpu().item()
            
            # Filter by confidence
            if score > 0.5:
                box = prediction[0]['boxes'][i].cpu().numpy().astype(np.int32)
                label_id = prediction[0]['labels'][i].cpu().item()
                label = self.class_names[label_id]
                
                # Get mask if available
                mask = None
                if 'masks' in prediction[0]:
                    if len(prediction[0]['masks']) > i:
                        mask = prediction[0]['masks'][i, 0].cpu().numpy() > 0.5
                
                objects.append({
                    'box': box,
                    'label': label,
                    'score': score,
                    'mask': mask
                })
        
        return objects

# Video style transfer class
class VideoStyleTransfer:
    def __init__(self, style_model_path, device=None, preserve_color=False):
        """Initialize the video style transfer model"""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load transformer network
        print(f"Loading style model: {style_model_path}")
        self.net = transformer.TransformerNetwork()
        self.net.load_state_dict(torch.load(style_model_path))
        self.net = self.net.to(self.device)
        self.net.eval()
        
        self.preserve_color = preserve_color
        self.prev_stylized = None
    
    def stylize_frame(self, frame, apply_temporal=True):
        """
        Stylize a single video frame
        
        Args:
            frame: OpenCV frame in BGR format
            apply_temporal: Whether to apply temporal consistency
            
        Returns:
            Stylized frame in BGR format
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        content_image = Image.fromarray(frame_rgb)
        
        # Apply style transfer
        content_tensor = utils.itot(content_image).to(self.device)
        
        with torch.no_grad():
            stylized_tensor = self.net(content_tensor)
        
        # Apply temporal consistency if needed
        if apply_temporal and self.prev_stylized is not None:
            stylized_tensor = (1 - TEMPORAL_WEIGHT) * stylized_tensor + TEMPORAL_WEIGHT * self.prev_stylized
        
        # Store current output for next frame
        self.prev_stylized = stylized_tensor.clone()
        
        # Convert back to image
        stylized_image = utils.ttoi(stylized_tensor.detach())
        
        # Convert back to BGR for OpenCV
        stylized_frame = cv2.cvtColor(np.array(stylized_image), cv2.COLOR_RGB2BGR)
        
        return stylized_frame

# Region-based style transfer class
class RegionBasedStyleTransfer:
    def __init__(self, style_foreground_path, style_background_path, device=None, 
                preserve_color_fg=False, preserve_color_bg=False):
        """Initialize the region-based style transfer model"""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for style transfer: {self.device}")
        
        # Load transformer networks
        if style_foreground_path is not None:
            print(f"Loading foreground style model: {style_foreground_path}")
            self.net_fg = transformer.TransformerNetwork()
            self.net_fg.load_state_dict(torch.load(style_foreground_path))
            self.net_fg = self.net_fg.to(self.device)
            self.net_fg.eval()
        else:
            print("Foreground will use original pixels (no style)")
            self.net_fg = None
        
        if style_background_path is not None:
            print(f"Loading background style model: {style_background_path}")
            self.net_bg = transformer.TransformerNetwork()
            self.net_bg.load_state_dict(torch.load(style_background_path))
            self.net_bg = self.net_bg.to(self.device)
            self.net_bg.eval()
        else:
            print("Background will use original pixels (no style)")
            self.net_bg = None
        
        # Initialize object detector
        self.detector = ObjectDetector(device)
        
        # Style settings
        self.preserve_color_fg = preserve_color_fg
        self.preserve_color_bg = preserve_color_bg
        
        # Temporal consistency
        self.prev_stylized_fg = None
        self.prev_stylized_bg = None
    
    def stylize_region(self, frame, is_foreground=True, apply_temporal=True):
        """Apply style transfer to a region of a frame"""
        # Select appropriate network and settings
        if is_foreground:
            net = self.net_fg
            prev_stylized = self.prev_stylized_fg
            preserve_color = self.preserve_color_fg
        else:
            net = self.net_bg
            prev_stylized = self.prev_stylized_bg
            preserve_color = self.preserve_color_bg
            
        # Check if the corresponding network exists
        if net is None:
            return frame  # Return original frame if no style network
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        content_image = Image.fromarray(frame_rgb)
        
        # Apply style transfer
        content_tensor = utils.itot(content_image).to(self.device)
        
        with torch.no_grad():
            stylized_tensor = net(content_tensor)
            
            # Apply temporal consistency if needed
            if apply_temporal and prev_stylized is not None:
                stylized_tensor = (1 - TEMPORAL_WEIGHT) * stylized_tensor + TEMPORAL_WEIGHT * prev_stylized
            
            # Store current output for next frame
            if is_foreground:
                self.prev_stylized_fg = stylized_tensor.clone()
            else:
                self.prev_stylized_bg = stylized_tensor.clone()
        
        # Convert back to image
        stylized_image = utils.ttoi(stylized_tensor.detach())
        
        # Convert back to BGR for OpenCV
        stylized_frame = cv2.cvtColor(np.array(stylized_image), cv2.COLOR_RGB2BGR)
        
        return stylized_frame
    
    def process_frame(self, frame, apply_temporal=True, target_classes=None):
        """Process a frame with region-based style transfer"""
        # Get original frame dimensions
        height, width = frame.shape[:2]
        
        # Detect objects
        objects = self.detector.detect_objects(frame)
        
        # Create foreground mask (initialize with zeros)
        fg_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Add detected objects to foreground mask if they match target classes
        for obj in objects:
            if target_classes is None or obj['label'] in target_classes:
                if obj['mask'] is not None:
                    # Use object mask
                    fg_mask = np.logical_or(fg_mask, obj['mask']).astype(np.uint8) * 255
                else:
                    # Use bounding box
                    x1, y1, x2, y2 = obj['box']
                    cv2.rectangle(fg_mask, (x1, y1), (x2, y2), 255, -1)
        
        # Dilate the mask slightly to avoid hard edges
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
        
        # Create background mask (inverse of foreground)
        bg_mask = 255 - fg_mask
        
        # Convert masks to 3-channel and proper scale for blending
        fg_mask_3c = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR) / 255.0
        bg_mask_3c = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Start with the original frame
        result = frame.copy()
        
        # Apply background style if provided
        if self.net_bg is not None:
            # Style the entire frame with background style
            bg_stylized = self.stylize_region(frame, is_foreground=False, apply_temporal=apply_temporal)
            # Only apply to background areas (where bg_mask is non-zero)
            result = result * (1.0 - bg_mask_3c) + bg_stylized * bg_mask_3c
        
        # Apply foreground style if provided
        if self.net_fg is not None:
            # Style the entire frame with foreground style
            fg_stylized = self.stylize_region(frame, is_foreground=True, apply_temporal=apply_temporal)
            # Only apply to foreground areas (where fg_mask is non-zero)
            result = result * (1.0 - fg_mask_3c) + fg_stylized * fg_mask_3c
        
        return result.astype(np.uint8)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        return len(self.active_connections)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        return len(self.active_connections)

    async def send_frame(self, websocket: WebSocket, frame):
        await websocket.send_text(frame)

manager = ConnectionManager()

# Function to decode base64 image from browser
def decode_base64_image(data_url):
    # Extract the base64 encoded data
    match = re.match(r'data:image/(jpeg|png|jpg);base64,(.+)', data_url)
    if not match:
        return None
    
    image_type, base64_data = match.groups()
    
    # Decode base64 data
    image_data = base64.b64decode(base64_data)
    
    # Convert to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img

@app.get("/")
async def get_index():
    # Create the index.html file if it doesn't exist
    if not os.path.exists("static/index.html"):
        with open("static/index.html", "w") as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url=/static/index.html">
</head>
<body>
    <p>Redirecting...</p>
</body>
</html>""")
    
    return FileResponse("static/index.html")

@app.get("/styles")
async def get_styles():
    return {"styles": list(STYLES.keys())}

@app.get("/classes")
async def get_classes():
    return {"classes": AVAILABLE_CLASSES}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = await manager.connect(websocket)
    print(f"Client {client_id} connected")
    
    # Initialize style processor
    processor = None
    last_frame_time = 0
    target_fps = 15  # Target FPS for processing
    frame_interval = 1.0 / target_fps
    options = None
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            if data.startswith("init:"):
                # Initialize with options
                options_raw = data[5:]  # Remove "init:" prefix
                options_dict = json.loads(options_raw)
                options = StyleOptions(**options_dict)
                
                # Setup processor based on mode
                if options.mode == "static":
                    processor = VideoStyleTransfer(
                        STYLES[options.bg_style], 
                        preserve_color=options.preserve_color_bg
                    )
                    print(f"Initialized static style transfer with {options.bg_style} style")
                else:  # semantic
                    fg_style_path = STYLES[options.fg_style] if options.fg_style else None
                    processor = RegionBasedStyleTransfer(
                        style_foreground_path=fg_style_path,
                        style_background_path=STYLES[options.bg_style],
                        preserve_color_fg=options.preserve_color_fg,
                        preserve_color_bg=options.preserve_color_bg
                    )
                    print(f"Initialized semantic style transfer with targets: {options.target_classes}")
                
                # Send acknowledgement
                await websocket.send_text("initialized")
                
            elif data.startswith("data:image/"):
                # Check if enough time has passed since last frame
                current_time = time.time()
                if (current_time - last_frame_time) < frame_interval:
                    # Skip this frame to maintain target FPS
                    continue
                
                last_frame_time = current_time
                
                # Process received frame
                if processor is not None and options is not None:
                    # Decode frame from base64
                    frame = decode_base64_image(data)
                    
                    if frame is not None:
                        # Process frame based on mode
                        start_time = time.time()
                        
                        if isinstance(processor, VideoStyleTransfer):
                            stylized = processor.stylize_frame(frame)
                        else:  # RegionBasedStyleTransfer
                            stylized = processor.process_frame(
                                frame, 
                                target_classes=options.target_classes
                            )
                        
                        process_time = time.time() - start_time
                        
                        # Add FPS indicator
                        fps = 1.0 / process_time if process_time > 0 else 0
                        cv2.putText(
                            stylized, 
                            f"FPS: {fps:.1f}", 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, 
                            (0, 255, 0), 
                            2
                        )
                        
                        # Convert to base64 for sending
                        _, buffer = cv2.imencode('.jpg', stylized, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                        
                        # Send processed frame back
                        await websocket.send_text(f"data:image/jpeg;base64,{jpg_as_text}")
                    else:
                        await websocket.send_text("error:invalid_frame_format")
                else:
                    await websocket.send_text("error:not_initialized")
                    
            elif data == "stop":
                # Release resources
                processor = None
                options = None
                await websocket.send_text("stopped")
    
    except WebSocketDisconnect:
        client_id = manager.disconnect(websocket)
        print(f"Client {client_id} disconnected")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)