#%%
import os
from dotenv import load_dotenv
os.environ["QWEN_2_5_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM2_ENABLED"] = "False"
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_GROUNDINGDINO_ENABLED"] = "False"
from inference import get_model

load_dotenv()

PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/20"

def ssi_bounding_box(video_path):
    
    print("Running SSI bounding box on:", video_path)
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    if ROBOFLOW_API_KEY is None:
        raise RuntimeError("ROBOFLOW_API_KEY not found in environment")
    
    PLAYER_DETECTION_MODEL = get_model(
        model_id=PLAYER_DETECTION_MODEL_ID,
        api_key=ROBOFLOW_API_KEY
    )

    return "Model loaded successfully"

# %%
