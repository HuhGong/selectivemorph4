from pspNet import *
import torch
import numpy as np
from PIL import Image
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
custom_size = (720, 720)
target_class_ids = [0]
# 클래스 이름 정의 (Pascal VOC 데이터셋)
class_names = {
    0: "background",
    # Stuff 클래스 (35개)
    1: "wall", 2: "sky", 3: "floor", 4: "ceiling", 5: "road",
    6: "grass", 7: "sidewalk", 8: "earth", 9: "mountain", 10: "water",
    11: "railroad", 12: "building", 13: "fence", 14: "rock", 15: "sand",
    16: "sea", 17: "river", 18: "plant", 19: "field", 20: "snow",
    21: "tree", 22: "bush", 23: "forest", 24: "hill", 25: "land",
    26: "staircase", 27: "runway", 28: "lake", 29: "waterfall", 30: "cloud",
    31: "flower", 32: "moss", 33: "path", 34: "stairs", 35: "ground",

    # Object 클래스 (115개)
    36: "person", 37: "car", 38: "chair", 39: "table", 40: "house",
    41: "door", 42: "window", 43: "lamp", 44: "cabinet", 45: "shelf",
    46: "mirror", 47: "carpet", 48: "curtain", 49: "pillow", 50: "screen",
    51: "bed", 52: "sofa", 53: "armchair", 54: "desk", 55: "wardrobe",
    56: "light", 57: "chandelier", 58: "fan", 59: "railing", 60: "column",
    61: "signboard", 62: "streetlight", 63: "box", 64: "boat", 65: "bus",
    66: "truck", 67: "bridge", 68: "tower", 69: "pole", 70: "bench",
    71: "bottle", 72: "bin", 73: "plant", 74: "flower pot", 75: "vase",
    76: "painting", 77: "book", 78: "bookcase", 79: "counter", 80: "sink",
    81: "toilet", 82: "clock", 83: "sculpture", 84: "cup", 85: "plate",
    86: "basket", 87: "tray", 88: "bowl", 89: "television", 90: "computer",
    91: "keyboard", 92: "phone", 93: "bag", 94: "shoe", 95: "umbrella",
    96: "bicycle", 97: "motorcycle", 98: "airplane", 99: "train", 100: "ship",
    101: "flag", 102: "banner", 103: "traffic light", 104: "sign", 105: "machine",
    106: "camera", 107: "mailbox", 108: "base", 109: "pot", 110: "barrel",
    111: "bucket", 112: "cart", 113: "stand", 114: "pipe", 115: "fountain",
    116: "candle", 117: "radiator", 118: "towel", 119: "map", 120: "blanket",
    121: "cloth", 122: "rack", 123: "case", 124: "handrail", 125: "sconce",
    126: "ashtray", 127: "paddle", 128: "ball", 129: "food", 130: "step",
    131: "tank", 132: "trade name", 133: "ashcan", 134: "pool", 135: "stool",
    136: "barrel", 137: "basket", 138: "waterfall", 139: "tent", 140: "minibike",
    141: "cradle", 142: "oven", 143: "ball", 144: "food", 145: "step",
    146: "tank", 147: "trade name", 148: "ashcan", 149: "pool", 150: "stool"
}


def resize_with_aspect_ratio_preserve_ids(image, target_size=custom_size):
    """
    이미지의 비율을 유지하면서 지정된 target_size 내에 맞게 리사이즈하고 패딩을 추가하며,
    클래스 ID 손상을 방지하기 위해 최근접 이웃 보간법을 사용합니다.
    """
    # 원본 이미지 크기
    width, height = image.size
    target_width, target_height = target_size

    # 비율 유지 리사이즈
    ratio = min(target_width / width, target_height / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # 최근접 이웃 보간법을 사용하여 리사이즈
    resized_image = image.resize((new_width, new_height), Image.NEAREST)

    # 패딩 추가 (가운데 정렬)
    delta_width = target_width - new_width
    delta_height = target_height - new_height
    padding = (delta_width // 2, delta_height // 2, delta_width - delta_width // 2, delta_height - delta_height // 2)

    # 검은색 배경으로 새 이미지를 생성
    padded_image = Image.new("P", target_size, 0)  # "P" 모드는 팔레트를 사용하는 모드
    padded_image.paste(resized_image, (padding[0], padding[1]))

    # 원래 팔레트 적용
    padded_image.putpalette(image.getpalette())

    return padded_image


def print_class_colors(image, palette, class_names):
    image_array = np.array(image)
    unique_classes = np.unique(image_array)

    print("\n사용된 클래스와 색상 정보:")
    for class_id in unique_classes:
        class_name = class_names.get(class_id, "Unknown")
        color = palette[class_id * 3: class_id * 3 + 3]
        print(f"Class ID {class_name} (ID: {class_id}): Color {tuple(color)}")


def image_loader(image, resize=None):
    # Open the image
    image = image.convert('RGB')

    # Resize the image if a resize value is provided
    if resize:
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    # Apply transformations and add batch dimension
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move to specified device
    return image.clone().detach().to(device, torch.float)  # clone().detach() 사용


def load_pspnet_model(weights_path):
    net = PSPNet(n_classes=150)
    state_dict = torch.load(weights_path, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net.to(device)


def setup_output_folder():
    output_folder = "output\\anno"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def clear_output_folder(output_folder):
    """output/anno 폴더의 모든 파일을 삭제합니다."""
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)  # 파일 삭제
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


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
    pspnet_weights_path = "weights\\pspnet50_ADE20K.pth"
    net = load_pspnet_model(pspnet_weights_path)

    return net


def setup_dataset():
    rootpath = "VOC2012\\"
    return make_datapath_list(rootpath)


def segment_image(net, content_img, anno_class_img, transform, device, content_img_width, content_img_height, p_palette,
                  output_folder):
    """이미지를 PSPNet을 사용하여 분할하고 클래스 이미지와 개별 클래스 마스크를 저장합니다."""
    phase = 'val'
    img, _ = transform(phase, content_img, anno_class_img)

    x = img.unsqueeze(0).to(device).float()

    with torch.no_grad():
        outputs = net(x)

    y = outputs[0][0].detach().cpu().numpy()
    y = np.argmax(y, axis=0)

    anno_class_img = Image.fromarray(np.uint8(y), mode="P")
    anno_class_img = anno_class_img.resize((content_img_width, content_img_height), Image.NEAREST)
    anno_class_img.putpalette(p_palette)

    output_path = os.path.join(output_folder, "anno_class_img.png")
    anno_class_img.save(output_path)

    print_class_colors(anno_class_img.convert("P"), p_palette, class_names)

    # 각 클래스별 이미지 저장
    unique_classes = np.unique(y)
    for class_id in unique_classes:
        class_mask = (y == class_id).astype(np.uint8) * 255  # 클래스 마스크 생성
        class_image = Image.fromarray(class_mask, mode="L")  # "L" 모드로 저장
        class_image = class_image.resize((content_img_width, content_img_height), Image.NEAREST)
        class_image_path = os.path.join(output_folder, f"anno_class_img_{class_id}.png")
        class_image.save(class_image_path)

    return anno_class_img


def main():
    output_folder = setup_output_folder()
    # output/anno 폴더 비우기
    clear_output_folder(output_folder)

    content_img, CONTENT_IMAGE_PATH = load_images()
    net = initialize_models(device)
    train_img_list, train_anno_list, val_img_list, val_anno_list = setup_dataset()

    # 임시 anno_class_img 로드 (실제로는 사용되지 않음)
    try:
        anno_file_path = val_anno_list[0]
        anno_class_img = Image.open(anno_file_path)
    except Exception as e:
        print(f"Error loading annotation image: {e}")
        exit(1)

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    transform = DataTransform(input_size=475,
                              color_mean=cnn_normalization_mean,
                              color_std=cnn_normalization_std)

    p_palette = anno_class_img.getpalette()

    anno_class_img = segment_image(net, content_img, anno_class_img, transform, device,
                                   content_img.size[0], content_img.size[1], p_palette, output_folder)

    print("Segmentation 완료.")


if __name__ == "__main__":
    main()
