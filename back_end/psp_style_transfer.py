from pspNet import *
from torchvision.models import vgg19, VGG19_Weights
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import os
import sys
import logging

# 로거 설정
logging.basicConfig(level=logging.INFO, encoding='utf-8')  # 로그 레벨 설정
logger = logging.getLogger(__name__)

# 전달된 인자들
selected_classes = sys.argv[1:]  # 공백으로 구분된 인자들

# 클래스 ID를 int로 변환
selected_classes = [int(cls_id) for cls_id in selected_classes if cls_id.isdigit()]

# 로그 남기기
logger.info("Selected classes: %s", selected_classes)  # 출력
logger.info("Type of selected_classes: %s", type(selected_classes))  # 출력

target_class_ids = selected_classes




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
custom_size = (720, 720)
# custom_size = (420, 420)



class_names = {
    0: "Background", 1: "Aeroplane", 2: "Bicycle", 3: "Bird", 4: "Boat", 5: "Bottle", 6: "Bus", 7: "Car",
    8: "Cat", 9: "Chair", 10: "Cow", 11: "Dining Table", 12: "Dog", 13: "Horse", 14: "Motorbike",
    15: "Person", 16: "Potted Plant", 17: "Sheep", 18: "Sofa", 19: "Train", 20: "TV Monitor"
}


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


def create_class_mask(segmentation_map, target_class_ids):
    """
    특정 클래스 ID에 대한 마스크를 생성합니다.
    """
    segmentation_array = np.array(segmentation_map)
    mask = np.isin(segmentation_array, target_class_ids).astype(np.uint8)  # 특정 클래스만 1로 설정
    return mask


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
    net = PSPNet(n_classes=21)
    state_dict = torch.load(weights_path, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net.to(device)


def setup_output_folder():
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def load_images():
    try:
        STYLE_IMAGE_PATH = "uploads\\styleImage\\styleImage.png"
        style_img = Image.open(STYLE_IMAGE_PATH).convert("RGB")

        CONTENT_IMAGE_PATH = "uploads\\contentImage\\contentImage.png"
        content_img = Image.open(CONTENT_IMAGE_PATH).convert("RGB")
    except Exception as e:
        print(f"Error loading images: {e}")
        exit(1)

    return style_img, content_img, CONTENT_IMAGE_PATH


def initialize_models(device):
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    pspnet_weights_path = "weights\\pspnet50_.pth"
    net = load_pspnet_model(pspnet_weights_path)

    return cnn, cnn_normalization_mean, cnn_normalization_std, net


def setup_dataset():
    rootpath = "VOC2012\\"
    return make_datapath_list(rootpath)


def process_segmentation(net, content_img, anno_class_img, transform, device, content_img_width, content_img_height,
                         p_palette):
    phase = 'val'
    img, anno_class_img = transform(phase, content_img, anno_class_img)

    x = img.unsqueeze(0).to(device).float()

    with torch.no_grad():
        outputs = net(x)

    y = outputs[0][0].detach().cpu().numpy()
    y = np.argmax(y, axis=0)

    anno_class_img = Image.fromarray(np.uint8(y), mode="P")
    anno_class_img = anno_class_img.resize((content_img_width, content_img_height), Image.NEAREST)
    anno_class_img.putpalette(p_palette)

    # 최종 결과 이미지만 저장하도록 수정
    return anno_class_img


def create_masked_content(anno_class_img, content_img, custom_size, target_class_ids, output_folder):
    resized_anno_class_img = resize_with_aspect_ratio_preserve_ids(anno_class_img, target_size=custom_size)
    segmentation_map = resized_anno_class_img.convert("P")
    class_mask = create_class_mask(segmentation_map, target_class_ids)

    resized_content_img = resize_with_aspect_ratio(content_img, target_size=custom_size)
    content_image_array = np.array(resized_content_img.convert("RGBA"))
    class_mask_expanded = np.stack([class_mask] * 4, axis=-1)

    masked_content = content_image_array * class_mask_expanded
    masked_content_image = Image.fromarray(masked_content)

    masked_content_path = os.path.join(output_folder, "masked_content.png")
    masked_content_image.save(masked_content_path)

    return masked_content_image, class_mask_expanded


def apply_style_transfer(masked_content_image, style_img, custom_size, cnn, cnn_normalization_mean,
                         cnn_normalization_std):
    content_img_resized = resize_with_aspect_ratio(masked_content_image, target_size=custom_size)
    content_image = image_loader(content_img_resized)

    style_img_resized = resize_with_aspect_ratio(style_img, target_size=custom_size)
    style_image = image_loader(style_img_resized)

    assert style_image.size() == content_image.size()

    input_img = content_image.clone()
    return run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                              content_image, style_image, input_img)


def save_final_result(style_transferred_output, CONTENT_IMAGE_PATH, class_mask_expanded, custom_size, output_folder):
    style_transferred_output_path = os.path.join(output_folder, 'imgtrans.png')
    save_image(style_transferred_output, style_transferred_output_path)

    style_transferred_img = Image.open(style_transferred_output_path).convert("RGBA")
    style_transferred_array = np.array(style_transferred_img)

    original_image = Image.open(CONTENT_IMAGE_PATH).convert("RGBA")
    resized_original_image = resize_with_aspect_ratio(original_image, target_size=custom_size)
    original_array = np.array(resized_original_image)

    result_array = np.where(class_mask_expanded, style_transferred_array, original_array)
    final_result = Image.fromarray(result_array.astype(np.uint8))

    # final_combined_image.png만 저장하도록 수정
    final_result_path = os.path.join(output_folder, "sample_7.png")
    final_result.save(final_result_path)
    return final_result_path


def main():
    output_folder = setup_output_folder()
    style_img, content_img, CONTENT_IMAGE_PATH = load_images()
    cnn, cnn_normalization_mean, cnn_normalization_std, net = initialize_models(device)
    train_img_list, train_anno_list, val_img_list, val_anno_list = setup_dataset()

    transform = DataTransform(input_size=475,
                              color_mean=cnn_normalization_mean,
                              color_std=cnn_normalization_std)
    try:
        anno_file_path = val_anno_list[0]
        anno_class_img = Image.open(anno_file_path)
    except Exception as e:
        print(f"Error loading annotation image: {e}")
        exit(1)

    p_palette = anno_class_img.getpalette()

    anno_class_img = process_segmentation(net, content_img, anno_class_img, transform, device,
                                          content_img.size[0], content_img.size[1], p_palette)

    print_class_colors(anno_class_img.convert("P"), p_palette, class_names)

    masked_content_image, class_mask_expanded = create_masked_content(anno_class_img, content_img,
                                                                      custom_size, target_class_ids, output_folder)

    style_transferred_output = apply_style_transfer(masked_content_image, style_img, custom_size,
                                                    cnn, cnn_normalization_mean, cnn_normalization_std)

    final_result_path = save_final_result(style_transferred_output, CONTENT_IMAGE_PATH,
                                          class_mask_expanded, custom_size, output_folder)

    print(f"최종 결과 이미지가 '{final_result_path}'에 저장되었습니다.")


if __name__ == "__main__":
    main()
