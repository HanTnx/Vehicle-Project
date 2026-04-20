Vehicle Detection, Classification & Traffic Analysis

## Project Structure

```
├── 1.1.vehicle_detection.ipynb          # Part 1.1: YOLOv26s vehicle detection training
├── 1.2.vehicle_classification.ipynb     # Part 1.2: CLIP ViT-L/14 classifier training + integrated pipeline
├── 2.adversarial_attacks.ipynb          # Part 2: Adversarial attacks (whitebox + blackbox + pipeline)
├── 3.VLM-integration-traffic-analysis.ipynb  # Part 3: Gemma 4 VLM traffic analysis via vLLM
│
├── detection_output_v2.mp4             # Output video with detection
├── detection_classifier_clip_output_v2.mp4             # Output video with detection and clip classifier  
│
├── adv_results_clip/                   # Adversarial attack outputs on CLIP classifier
│   ├── FGSM/                           #   Sample adversarial images (FGSM)
│   ├── PGD/                            #   Sample adversarial images (PGD)
│   ├── CW/                             #   Sample adversarial images (C&W)
│   ├── Square/                         #   Sample adversarial images (Square Attack)
│   ├── Pixle/                          #   Sample adversarial images (Pixle)
│   ├── OnePixel/                       #   Sample adversarial images (OnePixel)
│   └── report.txt                      #   Attack results summary
│
├── adv_pipeline_analysis/              # Bonus: pipeline-level attack transferability analysis
│   ├── analysis.json                   #   Quantitative results (det recall, cls flip rate, IoU)
│   └── <scenario>/                     #   Sample frames (clean vs adversarial vs diff)
│
├── output_gemma4_traffic/              # VLM analysis on segmented video (with SAM3 masks)
│   ├── traffic_analysis.json           #   Structured analysis per frame (31 frames)
│   └── analyzed_frames/                #   Sampled frames sent to Gemma 4
│
└── output_gemma4_traffic_no_segment/   # VLM analysis on raw video (without segmentation)
    ├── traffic_analysis.json
    └── analyzed_frames/
```

## Requirements

```bash
pip install torch torchvision transformers ultralytics torchattacks opencv-python pillow numpy pyyaml vllm requests
```

**Hardware:** NVIDIA GeForce RTX 4090 Ti (12 GB), CUDA 12.6, vLLM 0.19.0

## Part 1 — Object Detection and Classifier

### 1.1 Vehicle Detection (`1.1.vehicle_detection.ipynb`)

SAM3 auto-labels video frames with the text prompt `"vehicle"`, then YOLOv26s is fine-tuned on the generated dataset.

- **Model:** YOLOv26s (9.9M params) with COCO pretrained weights
- **Dataset:** 422 train / 120 valid / ~60 test images (auto-labeled, split 70/20/10)
- **Result:** mAP@0.5 = **95.6%**, Precision = 91.3%, Recall = 90.1%, Inference = 4.1 ms/image

### 1.2 Vehicle Type Classifier + Pipeline (`1.2.vehicle_classification.ipynb`)

CLIP ViT-L/14 is fine-tuned on two open-source datasets (Car-1000 + Vehicle-10) for 9-class vehicle type classification, then integrated with YOLOv26s into a single pipeline.

- **Model:** CLIP ViT-Large/patch14 (303.2M params)
- **Dataset:** 10,485 train / 900 valid / 450 test images
- **Classes:** Bus, MPV, SUV, Sedan, Sports Car, Truck, Van, Bicycle, Motorcycle
- **Result:** Validation accuracy = **91.33%**
- **Pipeline:** YOLOv26s detect → crop → CLIP classify → annotated output video (`detection_output_v2.mp4`)

## Part 2 — Adversarial Attack (`2.adversarial_attacks.ipynb`)

### Standalone Attacks

6 attacks on the CLIP classifier (ε = 8/255):

| Attack | Type | ASR |
|--------|------|-----|
| FGSM | Whitebox | 84% |
| PGD | Whitebox | **100%** |
| C&W | Whitebox | 94% |
| Square | Blackbox | **90%** |
| Pixle | Blackbox | 24% |
| OnePixel | Blackbox | 16% |

### Bonus: Pipeline Transferability

Frame-level attacks targeting CLIP caused YOLOv26s detection recall to drop from 100% to ~25–30%, demonstrating strong cross-model adversarial transferability.

## Part 3 — VLM Integration (`3.VLM-integration-traffic-analysis.ipynb`)

Gemma 4 E4B served locally via vLLM analyzes sampled frames from SAM3-segmented video.

- **Model:** `google/gemma-4-E4B-it` via vLLM (OpenAI-compatible API)
- **Input:** Segmented video with blue mask overlay on vehicles (SAM3)
- **Output:** Structured JSON per frame — traffic description, density rating, abnormal incidents
- **Result:** 31 frames analyzed, avg ~17 vehicles/frame, density = Medium, 0 abnormal incidents

## How to Run

1. **Detection:** Run `1.1.vehicle_detection.ipynb` — trains YOLOv26s
2. **Classification + Pipeline:** Run `1.2.vehicle_classification.ipynb` — trains CLIP classifier, generates `detection_output_v2.mp4`
3. **Adversarial Attacks:** Run `2.adversarial_attacks.ipynb` — generates attack results in `adv_results_clip/` and `adv_pipeline_analysis/`
4. **VLM Analysis:** Start vLLM server, then run `3.VLM-integration-traffic-analysis.ipynb`

```bash
# Start vLLM server (requires ~16GB VRAM)
vllm serve google/gemma-4-E4B-it --dtype auto --max-model-len 4096 --trust-remote-code --port 8000
```
