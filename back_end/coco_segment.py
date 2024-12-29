from torchvision import transforms
import math
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import bisenet.lib.data.transform_cv2 as T
from bisenet.lib.models import model_factory
from bisenet.configs import set_cfg_from_file
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
# Define constants
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()  #
torch.set_default_device(device)
custom_size = (512, 512)
target_class_ids = [0]

coco_class_names = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train",
                    8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign",
                    14: "parking meter", 15: "bench",
                    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
                    24: "zebra",
                    25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
                    34: "frisbee", 35: "skis",
                    36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
                    41: "skateboard",
                    42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
                    49: "knife",
                    50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
                    57: "carrot",
                    58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant",
                    65: "bed",
                    67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
                    77: "cell phone",
                    78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock",
                    86: "vase",
                    87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush"}

coco_palette = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
    (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
    (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
    (255, 99, 164), (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
    (134, 134, 103), (145, 148, 174), (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
    (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
    (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205), (147, 186, 208), (153, 69, 1),
    (3, 95, 161), (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88), (95, 32, 0),
    (130, 114, 135), (110, 129, 133), (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
    (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
    (246, 0, 122), (191, 162, 208)
]


def resize_with_aspect_ratio(image, target_size=custom_size):
    """
    이미지의 비율을 유지하면서 지정된 target_size 내에 맞게 리사이징하고 패딩을 추가합니다.
    """
    # 원본 이미지 크기
    width, height = image.size
    target_width, target_height = target_size

    # 비율 유지 리사이징
    ratio = min(target_width / width, target_height / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    resized_image = image.resize((new_width, new_height), Image.BILINEAR)

    # 패딩 추가 (가운데 정렬)
    delta_width = target_width - new_width
    delta_height = target_height - new_height
    padding = (delta_width // 2, delta_height // 2, delta_width - delta_width // 2, delta_height - delta_height // 2)

    padded_image = Image.new("RGBA", target_size, (0, 0, 0, 0))  # 검은 배경에 투명도 설정
    padded_image.paste(resized_image, (padding[0], padding[1]))

    return padded_image


def resize_with_aspect_ratio_preserve_ids(image, target_size=custom_size):
    width, height = image.size
    target_width, target_height = target_size

    ratio = min(target_width / width, target_height / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    resized_image = image.resize((new_width, new_height), Image.NEAREST)

    delta_width = target_width - new_width
    delta_height = target_height - new_height
    padding = (delta_width // 2, delta_height // 2, delta_width - delta_width // 2, delta_height - delta_height // 2)

    padded_image = Image.new("P", target_size, 0)
    padded_image.paste(resized_image, (padding[0], padding[1]))

    padded_image.putpalette(image.getpalette())

    return padded_image


def print_class_colors(image, palette, class_names):
    """
    세그멘테이션 이미지에서 사용된 COCO 클래스와 색상 정보를 출력하고,
    사용된 클래스 ID를 반환합니다.
    """
    image_array = np.array(image)
    unique_classes = np.unique(image_array)

    print("\n사용된 COCO 클래스와 색상 정보:")
    valid_classes = []
    for class_id in unique_classes:
        # 클래스 ID가 0이거나 coco_class_names에 포함된 경우만 처리
        if class_id == 0 or (class_id in class_names and class_id < len(palette)):
            class_name = class_names.get(class_id, "Background")
            color = tuple(int(c) for c in palette[class_id])
            print(f"Class ID {class_id} ({class_name}): Color {color}")
            valid_classes.append(class_id)  # 유효한 클래스 ID 저장
        else:
            print(f"Class ID {class_id}: Out of range or invalid")

    return valid_classes


def image_loader(image, resize=None):
    image = image.convert('RGB')

    if resize:
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    image = transform(image).unsqueeze(0)

    return image.clone().detach().to(device, torch.float)


def load_bisenet_coco_model(weights_path, config_path):
    cfg = set_cfg_from_file(config_path)
    net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')  # cfg.n_cats은 모델이 예측할 클래스 개수
    net.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True), strict=False)
    net.eval()
    net.cuda()  # 필요한가?
    return net.to(device)


def setup_output_folder():
    output_folder = "output\\anno"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def load_images():
    try:
        CONTENT_IMAGE_PATH = "uploads\\contentImage\\contentImage.png"
        content_img = Image.open(CONTENT_IMAGE_PATH).convert("RGB")
    except Exception as e:
        print(f"Error loading images: {e}")
        exit(1)

    return content_img, CONTENT_IMAGE_PATH


def initialize_models(device):
    bisenet_coco_weights_path = "weights\\model_final_v2_coco.pth"
    # bisenet_coco_config_path = "../../coco_segment/bisenet/configs/bisenetv2_coco.py"
    bisenet_coco_config_path = "bisenet/configs/bisenetv2_coco.py"
    net = load_bisenet_coco_model(bisenet_coco_weights_path, bisenet_coco_config_path)

    return net


def segment_image(net, content_img, transform, device, content_img_width, content_img_height, p_palette,
                  output_folder):
    p_palette = np.zeros((256, 3), dtype=np.uint8)
    for class_id, color in zip(coco_class_names.keys(), coco_palette):
        p_palette[class_id] = color

    # prepare data
    to_tensor = T.ToTensor(
        mean=(0.3257, 0.3690, 0.3223),  # city, rgb
        std=(0.2112, 0.2148, 0.2115),
    )
    im = np.array(content_img)[:, :, ::-1]  # BGR에서 RGB로 변환
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()  # 배치 차원 추가

    # shape divisor
    org_size = im.size()[2:]
    new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]

    # inference
    im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
    # 메모리 효율을 위해 즉시 처리
    out = net(im)[0]
    im = None  # 사용이 끝난 변수 즉시 해제
    torch.cuda.empty_cache()
    out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear')
    # segmentation_map = out.argmax(dim=1) # out 결과가 segmentation map 결과.
    segmentation_map = out.argmax(dim=1).squeeze().detach().cpu().numpy()
    out = None
    torch.cuda.empty_cache()

    # 모델 클래스 ID -> COCO 클래스 ID 매핑
    model_to_coco_id = {i: coco_id for i, coco_id in enumerate(coco_class_names.keys())}

    # segmentation_map_coco 생성 및 매핑
    segmentation_map_coco = np.full_like(segmentation_map, fill_value=0, dtype=np.uint8)  # 기본값 0으로 초기화
    for model_id, coco_id in model_to_coco_id.items():
        segmentation_map_coco[segmentation_map == model_id] = coco_id

    # None 값 확인 및 처리
    if np.any(segmentation_map_coco == None):
        raise ValueError("segmentation_map_coco contains invalid None values.")

    # segmentation_map 값 범위 확인 및 제한
    segmentation_map_coco = np.clip(segmentation_map_coco, 0, len(coco_class_names) - 1)

    # 팔레트를 세그멘테이션 맵에 적용
    segmentation_image = Image.fromarray(segmentation_map_coco.astype(np.uint8), mode="P")
    segmentation_image.putpalette(p_palette.flatten())
    # # segmentation_image.show()

    anno_class_img = Image.fromarray(segmentation_map_coco.astype(np.uint8), mode="P")
    anno_class_img = anno_class_img.resize((content_img_width, content_img_height), Image.NEAREST)
    anno_class_img.putpalette(p_palette.flatten())  # 팔레트 설정

    output_path = os.path.join(output_folder, "anno_class_img_coco.png")
    anno_class_img.save(output_path)
    valid_class_ids = print_class_colors(segmentation_map_coco, p_palette, coco_class_names)
    # print_class_colors(segmentation_map_coco, p_palette, coco_class_names)
    # unique_classes = np.unique(segmentation_map)
    for class_id in valid_class_ids:
        class_mask = (segmentation_map_coco == class_id).astype(np.uint8) * 255  # 클래스 마스크 생성
        class_image = Image.fromarray(class_mask, mode="L")  # "L" 모드로 저장
        class_image = class_image.resize((content_img_width, content_img_height), Image.NEAREST)
        class_image_path = os.path.join(output_folder, f"anno_class_img_{class_id}.png")
        class_image.save(class_image_path)
        print(f"Class ID {class_id}: 저장 완료 - {class_image_path}")

    return anno_class_img


def main():
    output_folder = setup_output_folder()
    content_img, CONTENT_IMAGE_PATH = load_images()
    net = initialize_models(device)

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    transform = transforms.Compose([
        transforms.Resize((475, 475)),  # 원하는 크기로 조정
        transforms.ToTensor(),  # 텐서로 변환
        transforms.Normalize(mean=cnn_normalization_mean, std=cnn_normalization_std)  # 정규화
    ])

    p_palette = np.zeros((256, 3), dtype=np.uint8)
    for class_id, color in zip(coco_class_names.keys(), coco_palette):
        p_palette[class_id] = color

    anno_class_img = segment_image(net, content_img, transform, device,
                                   content_img.size[0], content_img.size[1], p_palette, output_folder)

    print("Segmentation 완료.")


if __name__ == "__main__":
    main()
