
import torch
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PORT = 10003
 
FASTSAM_MODEL = "FastSAM-s.pt"
CLIP_MODEL    = "ViT-B-32"
CLIP_WEIGHTS  = "laion2b_s34b_b79k"
 
OPENROUTER_API_KEY = ""
VLM_MODEL = "openai/gpt-5.4-mini"
 
IMAGE_SIZE = 640
MIN_MASK_PIXELS = 300
 
THRESHOLD_UNKNOWN = 0.20
 
LABELS_PATH = "labels.json"
 
DEFAULT_LABELS = [
    "a person", "a car", "a bicycle", "a dog", "a cat",
    "a bottle", "a chair", "a phone", "a laptop",
    "a table", "a keyboard", "a monitor",
]
CLIP_PADDING = 20
