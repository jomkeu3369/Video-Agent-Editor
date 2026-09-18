import cv2
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from ultralytics import YOLO
from scenedetect import detect, ContentDetector
import os

import subprocess
import imageio_ffmpeg



OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
MAX_LOST_FRAMES = 15

MODEL_NAME = "ZhengPeng7/BiRefNet_HR"
INPUT_PATH = "src/data/input.mp4"
OUTPUT_PATH = "src/data/output.mp4"
TEMP_PATH = "src/data/temp_crop.mp4"

SCENE_THRESHOLD = 27.0

INPUT_SIZE = 1024
THRESHOLD = 0.7

FRAME_INTERVAL = 10
SMOOTH_ALPHA = 0.15

# 원본 높이의 85%만 사용해서 9:16 영역 생성
CROP_SCALE = 0.85

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


# =========================
# BiRefNet
# =========================

model = AutoModelForImageSegmentation.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = model.to(device)

if device == "cpu":
    model = model.float()

model.eval()

model_dtype = next(model.parameters()).dtype


transform_image = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================
# YOLO
# =========================

yolo = YOLO("yolo26n.pt")


# =========================
# Saliency Map
# =========================

def get_saliency_map(frame):
    height, width = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    input_tensor = transform_image(image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device=device, dtype=model_dtype)

    with torch.inference_mode():
        preds = model(input_tensor)

    if isinstance(preds, (list, tuple)):
        saliency = preds[-1]
    elif hasattr(preds, "logits"):
        saliency = preds.logits
    else:
        saliency = preds

    saliency = saliency.sigmoid().float().cpu()
    saliency = saliency[0].squeeze().numpy()

    saliency = cv2.resize(
        saliency,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    return saliency


# =========================
# Saliency fallback
# =========================

def get_saliency_center(saliency):
    mask = (saliency > THRESHOLD).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels <= 1:
        return None

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    ys, xs = np.where(labels == largest_label)

    if len(xs) == 0:
        return None

    center_x = int((xs.min() + xs.max()) / 2)
    center_y = int((ys.min() + ys.max()) / 2)

    return center_x, center_y


# =========================
# YOLO + Saliency
# =========================

def get_saliency_target(frame, boxes, track_ids):
    saliency = get_saliency_map(frame)

    candidates = []

    for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)

        roi = saliency[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        mean_score = float(roi.mean())
        high_score = float(np.percentile(roi, 90))

        score = mean_score * 0.7 + high_score * 0.3

        candidates.append((score, track_id))

    if len(candidates) == 0:
        return None

    candidates.sort(reverse=True)

    return candidates[0][1]


# =========================
# 영상 분석
# =========================

def analyze_video(path, scenes):
    cap = cv2.VideoCapture(path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scene_results = []

    for scene_idx, (start_frame, end_frame) in enumerate(scenes):
        points = []

        target_id = None
        lost_frames = 0
        last_center = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in tqdm(
            range(start_frame, end_frame),
            desc=f"Scene {scene_idx + 1}"
        ):
            ret, frame = cap.read()

            if not ret:
                break

            result = yolo.track(
                frame,
                persist=True,
                classes=[0],
                conf=0.25,
                tracker="botsort.yaml",
                verbose=False,
                device=0 if device == "cuda" else "cpu"
            )[0]

            if result.boxes.id is None:
                if last_center is not None:
                    points.append((
                        frame_idx,
                        last_center[0],
                        last_center[1]
                    ))

                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()


            # 처음에만 Saliency로 타겟 선정
            if target_id is None:
                target_id = get_saliency_target(
                    frame,
                    boxes,
                    track_ids
                )

                lost_frames = 0


            if target_id is None:
                continue


            # 타겟이 현재 프레임에서 사라짐
            if target_id not in track_ids:
                lost_frames += 1

                if last_center is not None:
                    points.append((
                        frame_idx,
                        last_center[0],
                        last_center[1]
                    ))

                # 일정 시간 이상 사라진 경우에만 재선정
                if lost_frames >= MAX_LOST_FRAMES:
                    target_id = None
                    lost_frames = 0

                continue


            # 타겟 발견
            lost_frames = 0

            index = track_ids.index(target_id)

            x1, y1, x2, y2 = map(
                int,
                boxes[index]
            )

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            last_center = (
                center_x,
                center_y
            )

            points.append((
                frame_idx,
                center_x,
                center_y
            ))


        scene_results.append({
            "start": start_frame,
            "end": end_frame,
            "points": points
        })

    cap.release()

    return scene_results, total_frames, width, height

def make_tracking_points(scene_results, total_frames, width, height):
    result = [
        (width / 2, height / 2)
        for _ in range(total_frames)
    ]

    for scene in scene_results:
        start_frame = scene["start"]
        end_frame = scene["end"]
        points = scene["points"]

        if len(points) == 0:
            continue

        frames = np.array([point[0] for point in points])
        xs = np.array([point[1] for point in points])
        ys = np.array([point[2] for point in points])

        scene_frames = np.arange(start_frame, end_frame)

        interp_x = np.interp(scene_frames, frames, xs)
        interp_y = np.interp(scene_frames, frames, ys)

        prev_x = interp_x[0]
        prev_y = interp_y[0]

        for i, frame_idx in enumerate(scene_frames):
            x = interp_x[i]
            y = interp_y[i]

            smooth_x = SMOOTH_ALPHA * x + (1 - SMOOTH_ALPHA) * prev_x
            smooth_y = SMOOTH_ALPHA * y + (1 - SMOOTH_ALPHA) * prev_y

            result[frame_idx] = (smooth_x, smooth_y)

            prev_x = smooth_x
            prev_y = smooth_y

    return result

# =========================
# 9:16 영역
# =========================

def get_crop_bbox(frame, center_x, center_y):
    height, width = frame.shape[:2]

    crop_height = int(height * CROP_SCALE)
    crop_width = int(crop_height * 9 / 16)

    if crop_width > width:
        crop_width = width
        crop_height = int(crop_width * 16 / 9)

    x1 = int(center_x - crop_width / 2)
    y1 = int(center_y - crop_height / 2)

    x1 = max(0, min(x1, width - crop_width))
    y1 = max(0, min(y1, height - crop_height))

    x2 = x1 + crop_width
    y2 = y1 + crop_height

    return x1, y1, x2, y2


# =========================
# 결과 영상
# =========================

def make_video(input_path, output_path, points):
    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT)
    )

    for frame_idx in tqdm(range(total_frames), desc="Render"):
        ret, frame = cap.read()

        if not ret:
            break

        center_x, center_y = points[frame_idx]

        x1, y1, x2, y2 = get_crop_bbox(
            frame,
            center_x,
            center_y
        )

        crop = frame[y1:y2, x1:x2]

        crop = cv2.resize(
            crop,
            (OUTPUT_WIDTH, OUTPUT_HEIGHT)
        )

        writer.write(crop)

    cap.release()
    writer.release()

def merge_audio(video_path, original_path, output_path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i", video_path,
        "-i", original_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path
    ]

    subprocess.run(command, check=True)

def detect_scenes(path):
    scene_list = detect(
        path,
        ContentDetector(threshold=SCENE_THRESHOLD)
    )

    scenes = []

    for start, end in scene_list:
        scenes.append((
            start.get_frames(),
            end.get_frames()
        ))

    print("Scenes:", len(scenes))

    return scenes

def main():
    scenes = detect_scenes(INPUT_PATH)

    scene_results, total_frames, width, height = analyze_video(
        INPUT_PATH,
        scenes
    )

    points = make_tracking_points(
        scene_results,
        total_frames,
        width,
        height
    )

    make_video(INPUT_PATH, TEMP_PATH, points)
    merge_audio(TEMP_PATH, INPUT_PATH, OUTPUT_PATH)

    if os.path.exists(TEMP_PATH):
        os.remove(TEMP_PATH)

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()