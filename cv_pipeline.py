# %%
import os
from dotenv import load_dotenv
from openai import batches
os.environ["QWEN_2_5_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM2_ENABLED"] = "False"
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_GROUNDINGDINO_ENABLED"] = "False"
from inference import get_model
from tqdm import tqdm
import supervision as sv
import torch
from transformers import AutoProcessor, SiglipVisionModel
import numpy as np
from more_itertools import chunked
import clip
from PIL import Image
import umap
from sklearn.cluster import KMeans
from sports.common.team import TeamClassifier


# commented out next 4 lines due to kernel craching. I originally trained the model in google colab with a GPU but this crashes in streamlit. instead of siglip, i will use clip.
#SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(DEVICE)
#EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH) commented out this line due to kernel crashing

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_MODEL, CLIP_PREPROCESS = clip.load(
    "ViT-B/32",
    device=DEVICE
)

# pip install git+https://github.com/roboflow/sports.git

load_dotenv()

PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/20"

def ssi_bounding_box(video_path, target_video_path):
    
    print("Running SSI bounding box on:", video_path)
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    if ROBOFLOW_API_KEY is None:
        raise RuntimeError("ROBOFLOW_API_KEY not found in environment")
    
    PLAYER_DETECTION_MODEL = get_model(
        model_id=PLAYER_DETECTION_MODEL_ID,
        api_key=ROBOFLOW_API_KEY
    )
    
    STRIDE = 30 # use a larger stride for faster inferences during testing
    BALL_ID = 0
    GOALKEEPER_ID = 1
    PLAYER_ID = 2
    REFEREE_ID = 3

    
    def extract_crops(video_path: str):
        frame_generator = sv.get_video_frames_generator(video_path, stride=STRIDE)

        crops=[]
        for frame in tqdm(frame_generator, desc='collecting crops'):
            result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
            detections = sv.Detections.from_inference(result)
            detections = detections.with_nms(threshold=0.5, class_agnostic=True)
            detections = detections[detections.class_id== PLAYER_ID]
            crops += [
                sv.crop_image(frame, xyxy)
                for xyxy
                in detections.xyxy
            ]
        return crops
    
    crops = extract_crops(video_path)
    print("Number of crops extracted:", len(crops))
    sv.plot_images_grid(crops[:100], grid_size=(10,10))
    
    BATCH_SIZE = 32
    crops = [sv.cv2_to_pillow(crop) for crop in crops]
    batches = chunked(crops, BATCH_SIZE)
    data = []

    # commented out the next block due to kernel crashing. originally used siglip model but switched to clip model
 #   with torch.no_grad():
 #       for batch in tqdm(batches, desc='embeddings extraction'):
 #           inputs = EMBEDDINGS_PROCESSOR(images=batch, return_tensors='pt').to(DEVICE)
 #           outputs = EMBEDDINGS_MODEL(**inputs)
 #           embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
 #           data.append(embeddings)

    with torch.no_grad():
        for batch in tqdm(batches, desc="embeddings extraction"):

            processed_images = []

            for img in batch:
                if isinstance(img, Image.Image):
                    pil_img = img
                else:
                    pil_img = Image.fromarray(img)
            
                processed_images.append(CLIP_PREPROCESS(pil_img))

            images = torch.stack(processed_images).to(DEVICE)

            embeddings = CLIP_MODEL.encode_image(images)
            embeddings = embeddings.cpu().numpy()

            data.append(embeddings)

            print("Embedding batch shape:", embeddings.shape)

    data = np.concatenate(data)

    data.shape
    print("data shape is:",data.shape)


    REDUCER = umap.UMAP(n_components=3)
    CLUSTERING_MODEL = KMeans(n_clusters=2)

    #debugging
    print("DEBUG — type(data):", type(data))
    print("DEBUG — data.shape BEFORE any fix:", np.array(data).shape)

    # dimensionality reduction
    projections = REDUCER.fit_transform(data)
    print("Projections shape:", projections.shape)

    clusters = CLUSTERING_MODEL.fit_predict(projections)

    clusters[:10]

    team_0 = [
    crop
    for crop, cluster
    in zip(crops, clusters)
    if cluster ==0
    ]

    sv.plot_images_grid(team_0[:100], grid_size=(10, 10))

    def resolve_goalkeepers_team_id(players_detections: sv.Detections,
                                goalkeepers_detections: sv.Detections
                                ):

        goalkeepers_xy = goalkeepers_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

        team_0_centroid = players_xy[players_detections.class_id ==0].mean(axis=0)
        team_1_centroid = players_xy[players_detections.class_id ==1].mean(axis=0)

        goalkeepers_team_ids = []
        for goalkeeper_xy in goalkeepers_xy:
            dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
            dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
            goalkeepers_team_ids.append(0 if dist_0 < dist_1 else 1)

        return np.array(goalkeepers_team_ids)
    
    ellipse_annotator = sv.EllipseAnnotator(
        color = sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        thickness = 2
    )

    label_annotator = sv.LabelAnnotator(
        color = sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=20, height=17
    )

    tracker = sv.ByteTrack()
    tracker.reset()

    print("Processing video for bounding box annotation...")

    video_info = sv.VideoInfo.from_video_path(video_path)
    video_sink = sv.VideoSink(target_video_path, video_info=video_info)
    frame_generator = sv.get_video_frames_generator(video_path)
    frame = next(frame_generator)
    
    with video_sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):

            result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
            detections = sv.Detections.from_inference(result)

            ball_detections = detections[detections.class_id == BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            all_detections = detections[detections.class_id != BALL_ID]
            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
            all_detections = tracker.update_with_detections(all_detections)

            players_detections = all_detections[all_detections.class_id == PLAYER_ID]
            goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
            referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            players_crops = [sv.cv2_to_pillow(crop) for crop in players_crops]
            BATCH_SIZE = 32
            batches = chunked(players_crops, BATCH_SIZE)
            data = []


            #problem here

    # commented out the next block due to kernel crashing. originally used siglip model but switched to clip model
 #   with torch.no_grad():
 #       for batch in tqdm(batches, desc='embeddings extraction'):
 #           inputs = EMBEDDINGS_PROCESSOR(images=batch, return_tensors='pt').to(DEVICE)
 #           outputs = EMBEDDINGS_MODEL(**inputs)
 #           embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
 #           data.append(embeddings)

            with torch.no_grad():
                for batch in tqdm(batches, desc="embeddings extraction"):

                    processed_images = []

                    for img in batch:
                        if isinstance(img, Image.Image):
                            pil_img = img
                        else:
                            pil_img = Image.fromarray(img)
                    
                        processed_images.append(CLIP_PREPROCESS(pil_img))

                    images = torch.stack(processed_images).to(DEVICE)

                    embeddings = CLIP_MODEL.encode_image(images)
                    embeddings = embeddings.cpu().numpy()

                    data.append(embeddings)

                    #print("Embedding batch shape:", embeddings.shape)

            data = np.concatenate(data)

            #data.shape
            #print("data shape is:",data.shape)


            REDUCER = umap.UMAP(n_components=3)
            CLUSTERING_MODEL = KMeans(n_clusters=2)

            #debugging
            #print("DEBUG — type(data):", type(data))
            #print("DEBUG — data.shape BEFORE any fix:", np.array(data).shape)

            # dimensionality reduction
            projections = REDUCER.fit_transform(data)
            print("Projections shape:", projections.shape)

            clusters = CLUSTERING_MODEL.fit_predict(projections)


            ###
            #players_detections.class_id = CLUSTERING_MODEL.fit_predict(players_crops) # worked with siglip

            goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
                players_detections, goalkeepers_detections
                )

            referees_detections.class_id -= 1

            all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])
            all_detections.class_id = all_detections.class_id.astype(int) # added this to ensure all_detections were integers


            labels = [
                f"{tracker_id}"
                for tracker_id
                in all_detections.tracker_id
                ]

            annotated_frame = frame.copy()
            annotated_frame = ellipse_annotator.annotate(annotated_frame, all_detections)
            annotated_frame = triangle_annotator.annotate(annotated_frame, ball_detections)
            annotated_frame = label_annotator.annotate(annotated_frame, all_detections, labels=labels)

            video_sink.write_frame(annotated_frame)

    sv.plot_image(annotated_frame)


    



# %%

