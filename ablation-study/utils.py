# utils.py

import os
import json
import torch
from transformers import CLIPProcessor, CLIPModel

def process_threshold_combination(args, extract_keyshots, JSON_OUTPUT_DIR, SEQUENCES_DIR):
    hist_thresh, siglip_thresh = args

    # Initialize Model and Processor inside the Process
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print(f"\n Running with Hist_Threshold={hist_thresh} | SigLIP_Threshold={siglip_thresh}")

    keyshots = []

    for seq in os.listdir(SEQUENCES_DIR):
        seq_path = os.path.join(SEQUENCES_DIR, seq)
        if not os.path.isdir(seq_path):
            continue

        seq_keyshots = extract_keyshots(seq_path, hist_thresh, siglip_thresh, device, processor, model)
        keyshots.extend(seq_keyshots)

    json_path = os.path.join(JSON_OUTPUT_DIR, f"optimized_results_hist-{hist_thresh}_siglip-{siglip_thresh}.json")
    with open(json_path, "w") as json_file:
        json.dump({"hist_thresh": hist_thresh, "siglip_thresh": siglip_thresh, "keyshots": keyshots}, json_file, indent=4)

    print(f"JSON saved at: {json_path}")