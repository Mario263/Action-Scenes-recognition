
import os
import cv2
import json
import numpy as np
import shutil
import torch
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from scenedetect import detect, ThresholdDetector, ContentDetector
from multiprocessing import Pool, set_start_method
import tempfile

from utils import process_threshold_combination

os.environ['TMPDIR'] = tempfile.gettempdir()


VIDEO_PATH = "/Users/mario/Desktop/Desktop/UofA/4.Winter-2025/ece-910/Wild-Bunch-output/TheWildBunch(1969).mp4"
FRAMES_DIR = "/Users/mario/Desktop/Desktop/UofA/4.Winter-2025/ece-910/Wild-Bunch-output/Frames-extraction"
SEQUENCES_DIR = "/Users/mario/Desktop/Desktop/UofA/4.Winter-2025/ece-910/Wild-Bunch-output/histogram-output-sequences"
JSON_OUTPUT_DIR = "/Users/mario/Desktop/Desktop/UofA/4.Winter-2025/ece-910/Wild-Bunch-output/json-output"


os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(SEQUENCES_DIR, exist_ok=True)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)


FPS = 60
print(f" FPS of the movie: {FPS}")

# Step 1: Extract and Resize Frames
def extract_frames(video_path, output_dir, resize_width=224, resize_height=224):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    print("\n Extracting and resizing frames...")
    for _ in tqdm(range(total_frames), desc="Extracting Frames"):
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (resize_width, resize_height))
        frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, resized_frame)
        frame_count += 1

    cap.release()
    print(f" Extracted and resized {frame_count} frames from the video.")

# Step 2: Detect Hard and Gradual Cut Boundaries
def detect_shot_boundaries(video_path):
    hard_cut_scenes = detect(video_path, ThresholdDetector(threshold=30.0, min_scene_len=15))
    hard_boundaries = [(scene[0].get_frames(), scene[1].get_frames()) for scene in hard_cut_scenes]
    print(f" Detected {len(hard_boundaries)} hard-cut sequences.")

    gradual_cut_scenes = detect(video_path, ContentDetector(threshold=30.0, min_scene_len=15))
    gradual_boundaries = [(scene[0].get_frames(), scene[1].get_frames()) for scene in gradual_cut_scenes]
    print(f"Detected {len(gradual_boundaries)} gradual-cut sequences.")

    return hard_boundaries, gradual_boundaries

#  Step 3: Organize Frames into Sequences and Save Metadata
def organize_frames_by_sequence(frame_dir, boundaries, cut_type):
    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    sequences_metadata = {"sequences": []}

    for i, (start, end) in enumerate(boundaries, 1):
        seq_dir = os.path.join(SEQUENCES_DIR, f"{cut_type}_sequence_{i}")
        os.makedirs(seq_dir, exist_ok=True)

        for frame in frames:
            frame_num = int(frame.split(".")[0].replace("frame_", ""))
            if start <= frame_num <= end:
                shutil.copy(os.path.join(frame_dir, frame), os.path.join(seq_dir, frame))

        sequences_metadata["sequences"].append({
            "sequence_id": i,
            "start_frame": start,
            "end_frame": end,
            "start_time": start / FPS,
            "end_time": end / FPS,
            "duration": (end - start) / FPS
        })

    metadata_json_path = os.path.join(JSON_OUTPUT_DIR, f"{cut_type}_metadata.json")
    with open(metadata_json_path, "w") as json_file:
        json.dump(sequences_metadata, json_file, indent=4)

    print(f" {cut_type.capitalize()} sequence metadata saved at: {metadata_json_path}")
    return sequences_metadata

#  Step 4: Compute Motion using Optical Flow
def compute_motion(image1, image2):
    if image1 is None or image2 is None:
        return 0

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape) == 3 else image2

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return np.mean(np.abs(flow))

#  Function to Extract Image Embeddings using CLIP
def get_image_embedding(image_path, device, processor, model):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.squeeze()

#  Step 5: Extract Keyshots
def extract_keyshots(sequence_dir, hist_threshold, siglip_threshold, device, processor, model):
    frames = sorted([f for f in os.listdir(sequence_dir) if f.endswith(".jpg")])
    keyshots = []
    prev_hist = None
    prev_embedding = None
    prev_frame = None

    for frame in frames:
        frame_path = os.path.join(sequence_dir, frame)
        image = cv2.imread(frame_path)
        if image is None:
            continue

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is None:
            keyshots.append(frame_path)
            prev_hist = hist
            prev_embedding = get_image_embedding(frame_path, device, processor, model)
            prev_frame = gray_image
            continue

        hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        motion_score = compute_motion(prev_frame, gray_image)

        if hist_diff > hist_threshold:
            curr_embedding = get_image_embedding(frame_path, device, processor, model)
            similarity = torch.nn.functional.cosine_similarity(prev_embedding, curr_embedding, dim=0)

            if similarity < siglip_threshold or motion_score > 2.0:
                keyshots.append(frame_path)
                prev_embedding = curr_embedding
                prev_frame = gray_image

        prev_hist = hist

    return keyshots

#  Step 7: Run Experiments in Parallel using Multiprocessing
def run_experiments():
    extract_frames(VIDEO_PATH, FRAMES_DIR)
    hard_boundaries, gradual_boundaries = detect_shot_boundaries(VIDEO_PATH)

    organize_frames_by_sequence(FRAMES_DIR, hard_boundaries, "hard_cuts")
    organize_frames_by_sequence(FRAMES_DIR, gradual_boundaries, "gradual_cuts")

    hist_thresholds = [0.2, 0.3, 0.4, 0.5]
    siglip_thresholds = [0.65, 0.70, 0.75, 0.8, 0.85, 0.9]
    threshold_combinations = [(h, s) for h in hist_thresholds for s in siglip_thresholds]

    # Prepare arguments for process_threshold_combination
    args_list = [(h, s) for h, s in threshold_combinations]

    with Pool() as pool:
        pool.starmap(process_threshold_combination, [(args, extract_keyshots, JSON_OUTPUT_DIR, SEQUENCES_DIR) for args in args_list])

    print("All experiments completed.")

#  Main Entry Point
if __name__ == "__main__":
    # Explicitly set the start method to 'spawn'
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass  # start method already set

    run_experiments()