import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from tqdm import tqdm


# =========================
# 설정
# =========================

MODEL_NAME = "ZhengPeng7/BiRefNet_HR"

OUTPUT_DIR = "src/data/BiRefNet_HR_ROI"
CROP_DIR = os.path.join(
    OUTPUT_DIR,
    "crops"
)

THRESHOLD = 0.7
KERNEL_SIZE = 5

IMAGE_COUNT = 100
INPUT_SIZE = 1024

# Bounding Box 여백
PADDING_RATIO = 0.1


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    CROP_DIR,
    exist_ok=True
)


# =========================
# Saliency 후처리
# =========================

def postprocess_saliency(
    saliency_tensor,
    threshold=0.7,
    kernel_size=5
):

    saliency_np = saliency_tensor.cpu().numpy()

    raw_mask = (
        saliency_np * 255
    ).astype(
        np.uint8
    )


    # =========================
    # Threshold
    # =========================

    threshold_value = int(
        threshold * 255
    )

    _, binary_mask = cv2.threshold(
        raw_mask,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )


    # =========================
    # Morphology
    # =========================

    kernel = np.ones(
        (
            kernel_size,
            kernel_size
        ),
        np.uint8
    )


    opened_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel
    )


    cleaned_mask = cv2.morphologyEx(
        opened_mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # =========================
    # Largest Component
    # =========================

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            cleaned_mask,
            connectivity=8
        )
    )


    if num_labels <= 1:
        return (
            raw_mask,
            binary_mask,
            cleaned_mask
        )


    largest_label = (
        1
        + np.argmax(
            stats[
                1:,
                cv2.CC_STAT_AREA
            ]
        )
    )


    clean_mask = np.zeros_like(
        cleaned_mask
    )


    clean_mask[
        labels == largest_label
    ] = 255


    return (
        raw_mask,
        binary_mask,
        clean_mask
    )


# =========================
# Bounding Box 계산
# =========================

def get_bounding_box(
    mask,
    padding_ratio=0.1
):

    ys, xs = np.where(
        mask > 0
    )


    # foreground 없음
    if len(xs) == 0:
        return None


    x1 = xs.min()
    x2 = xs.max()

    y1 = ys.min()
    y2 = ys.max()


    width = x2 - x1
    height = y2 - y1


    # =========================
    # Padding
    # =========================

    padding_x = int(
        width * padding_ratio
    )

    padding_y = int(
        height * padding_ratio
    )


    image_height, image_width = (
        mask.shape
    )


    x1 = max(
        0,
        x1 - padding_x
    )

    y1 = max(
        0,
        y1 - padding_y
    )

    x2 = min(
        image_width - 1,
        x2 + padding_x
    )

    y2 = min(
        image_height - 1,
        y2 + padding_y
    )


    return (
        x1,
        y1,
        x2,
        y2
    )


# =========================
# Dataset
# =========================

dataset = load_dataset(
    "microsoft/cats_vs_dogs",
    split="train"
)


images = []


for data in dataset:

    if data["labels"] == 0:

        images.append(
            data[
                "image"
            ].convert(
                "RGB"
            )
        )


        if len(images) >= IMAGE_COUNT:
            break


# =========================
# Device
# =========================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "device:",
    device
)

print(
    "model:",
    MODEL_NAME
)


# =========================
# Model
# =========================

model = (
    AutoModelForImageSegmentation
    .from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
)


model = model.to(
    device
)


if device == "cpu":
    model = model.float()


model.eval()


model_dtype = next(
    model.parameters()
).dtype


print(
    "model dtype:",
    model_dtype
)


# =========================
# Transform
# =========================

transform_image = transforms.Compose([

    transforms.Resize(
        (
            INPUT_SIZE,
            INPUT_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [
            0.485,
            0.456,
            0.406
        ],

        [
            0.229,
            0.224,
            0.225
        ]
    )

])


# =========================
# Inference
# =========================

for idx, image in enumerate(

    tqdm(
        images,
        desc="Saliency ROI Detection"
    ),

    start=1
):


    # =========================
    # Preprocess
    # =========================

    input_tensor = transform_image(
        image
    )


    input_tensor = (
        input_tensor
        .unsqueeze(0)
        .to(
            device=device,
            dtype=model_dtype
        )
    )


    # =========================
    # Inference
    # =========================

    with torch.inference_mode():

        preds = model(
            input_tensor
        )


    # =========================
    # Output
    # =========================

    if isinstance(
        preds,
        (list, tuple)
    ):

        saliency = preds[-1]


    elif hasattr(
        preds,
        "logits"
    ):

        saliency = (
            preds.logits
        )


    else:

        saliency = preds


    saliency = (
        saliency
        .sigmoid()
        .cpu()
    )


    saliency = (
        saliency[0]
        .squeeze()
    )


    # =========================
    # Post Processing
    # =========================

    raw_mask, binary_mask, clean_mask = (
        postprocess_saliency(
            saliency,
            threshold=THRESHOLD,
            kernel_size=KERNEL_SIZE
        )
    )


    # =========================
    # 원본 이미지 크기 복원
    # =========================

    width, height = (
        image.size
    )


    raw_mask = cv2.resize(
        raw_mask,
        (
            width,
            height
        )
    )


    clean_mask = cv2.resize(
        clean_mask,
        (
            width,
            height
        ),
        interpolation=cv2.INTER_NEAREST
    )


    # =========================
    # Bounding Box
    # =========================

    bbox = get_bounding_box(
        clean_mask,
        padding_ratio=PADDING_RATIO
    )


    # Saliency 실패한 경우
    if bbox is None:

        print(
            f"\n[{idx}] "
            f"Bounding Box detection failed"
        )

        continue


    x1, y1, x2, y2 = bbox


    # =========================
    # ROI Crop
    # =========================

    roi_image = image.crop(
        (
            x1,
            y1,
            x2,
            y2
        )
    )


    # Crop 별도 저장
    roi_image.save(
        os.path.join(
            CROP_DIR,
            f"crop_{idx}.png"
        )
    )


    # =========================
    # Bounding Box 이미지
    # =========================

    bbox_image = np.array(
        image
    ).copy()


    # RGB이므로 matplotlib 출력용
    cv2.rectangle(
        bbox_image,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        (
            255,
            0,
            0
        ),
        4
    )


    # =========================
    # Visualization
    # =========================

    plt.figure(
        figsize=(20, 5)
    )


    # Original
    plt.subplot(
        1,
        4,
        1
    )

    plt.title(
        "Original"
    )

    plt.imshow(
        image
    )

    plt.axis(
        "off"
    )


    # Saliency
    plt.subplot(
        1,
        4,
        2
    )

    plt.title(
        "Saliency Mask"
    )

    plt.imshow(
        clean_mask,
        cmap="gray"
    )

    plt.axis(
        "off"
    )


    # Bounding Box
    plt.subplot(
        1,
        4,
        3
    )

    plt.title(
        "ROI Bounding Box"
    )

    plt.imshow(
        bbox_image
    )

    plt.axis(
        "off"
    )


    # Cropped
    plt.subplot(
        1,
        4,
        4
    )

    plt.title(
        "Saliency ROI Crop"
    )

    plt.imshow(
        roi_image
    )

    plt.axis(
        "off"
    )


    plt.tight_layout()


    # =========================
    # Save
    # =========================

    save_path = os.path.join(
        OUTPUT_DIR,
        f"saliency_roi_{idx}.png"
    )


    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


print(
    "Done"
)

print(
    f"Results: {OUTPUT_DIR}"
)

print(
    f"Crops: {CROP_DIR}"
)