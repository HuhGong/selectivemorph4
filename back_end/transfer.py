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
logging.basicConfig(level=logging.INFO, encoding='utf-8')
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
custom_size = (720, 720)

def resize_with_aspect_ratio(image, target_size=custom_size):
    width, height = image.size
    target_width, target_height = target_size

    ratio = min(target_width / width, target_height / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    resized_image = image.resize((new_width, new_height), Image.BILINEAR)

    delta_width = target_width - new_width
    delta_height = target_height - new_height
    padding = (delta_width // 2, delta_height // 2, delta_width - delta_width // 2, delta_height - delta_height // 2)
    padded_image = Image.new("RGBA", target_size, (0, 0, 0, 0))
    padded_image.paste(resized_image, (padding[0], padding[1]))

    return padded_image

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
        logger.error(f"Error loading images: {e}")
        sys.exit(1)

    return style_img, content_img, CONTENT_IMAGE_PATH

def initialize_models(device):
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    return cnn, cnn_normalization_mean, cnn_normalization_std

def apply_style_transfer(content_img, style_img, custom_size, cnn, cnn_normalization_mean, cnn_normalization_std):
    content_img_resized = resize_with_aspect_ratio(content_img, target_size=custom_size)
    content_image = image_loader(content_img_resized)

    style_img_resized = resize_with_aspect_ratio(style_img, target_size=custom_size)
    style_image = image_loader(style_img_resized)

    assert style_image.size() == content_image.size()
    input_img = content_image.clone()

    return run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                              content_image, style_image, input_img)

def save_final_result(style_transferred_output, output_folder):
    final_result_path = os.path.join(output_folder, "final_combined_image.png")
    save_image(style_transferred_output, final_result_path)
    return final_result_path

def main():
    try:
        output_folder = setup_output_folder()
        style_img, content_img, CONTENT_IMAGE_PATH = load_images()
        cnn, cnn_normalization_mean, cnn_normalization_std = initialize_models(device)

        style_transferred_output = apply_style_transfer(content_img, style_img,
                                                        custom_size, cnn,
                                                        cnn_normalization_mean,
                                                        cnn_normalization_std)

        final_result_path = save_final_result(style_transferred_output, output_folder)
        logger.info(f"최종 결과 이미지가 '{final_result_path}'에 저장되었습니다.")

    except Exception as e:
        logger.error(f"Error during style transfer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
