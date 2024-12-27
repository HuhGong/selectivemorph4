from pspNet import *
from torchvision.models import vgg19, VGG19_Weights
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
custom_size = (720, 720)


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


def setup_output_folder():
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def load_images():
    try:
        style_image_path = "uploads\\styleImage\\styleImage.png"
        style_img = Image.open(style_image_path).convert("RGB")

        content_image_path = "uploads\\contentImage\\contentImage.png"
        content_img = Image.open(content_image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading images: {e}")
        exit(1)

    return style_img, content_img, content_image_path


def initialize_models(device):
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    return cnn, cnn_normalization_mean, cnn_normalization_std


def apply_style_transfer(content_image, style_image, cnn, cnn_normalization_mean,
                         cnn_normalization_std, custom_size):
    content_image = image_loader(content_image, custom_size)
    style_image = image_loader(style_image, custom_size)

    assert style_image.size() == content_image.size()

    input_img = content_image.clone()
    return run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                              content_image, style_image, input_img)


def save_final_result(style_transferred_output, output_folder):
    # 스타일 전송된 이미지를 바로 최종 결과로 저장
    final_result_path = os.path.join(output_folder, "final_combined_image.png")
    save_image(style_transferred_output, final_result_path)

    return final_result_path


def main():
    output_folder = setup_output_folder()
    style_img, content_img, content_image_path = load_images()
    cnn, cnn_normalization_mean, cnn_normalization_std = initialize_models(device)

    style_transferred_output = apply_style_transfer(content_img, style_img,
                                                    cnn, cnn_normalization_mean, cnn_normalization_std, custom_size)

    final_result_path = save_final_result(style_transferred_output,
                                          output_folder)

    print(f"최종 결과 이미지가 '{final_result_path}'에 저장되었습니다.")


if __name__ == "__main__":
    main()
