import os
import torch
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# 이미지 로드 및 전처리
def load_image(image_path, size=(720, 720)):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # 배치 차원 추가
    return image

# 이미지 후처리
def deprocess_image(tensor):
    tensor = tensor.squeeze(0)  # 배치 차원 제거
    tensor = tensor.detach().cpu().numpy()
    tensor = np.clip(tensor, 0, 1)
    tensor = tensor.transpose(1, 2, 0)  # 채널 차원 변경
    return (tensor * 255).astype(np.uint8)

# 스타일 전이 함수
def style_transfer(content_img, style_img, num_steps=1000, style_weight=1000000, content_weight=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # VGG19 모델 로드
    vgg = models.vgg19(pretrained=True).features.to(device).eval()

    # 콘텐츠와 스타일 손실을 계산하는 함수
    def get_features(image):
        features = []
        for layer in vgg.children():
            image = layer(image)
            features.append(image)
        return features

    content_features = get_features(content_img)
    style_features = get_features(style_img)

    # 생성할 이미지 초기화
    generated_img = content_img.clone().requires_grad_(True).to(device)

    optimizer = optim.Adam([generated_img], lr=0.003)

    for step in range(num_steps):
        optimizer.zero_grad()

        generated_features = get_features(generated_img)

        # 콘텐츠 손실
        content_loss = content_weight * torch.mean((generated_features[-1] - content_features[-1]) ** 2)

        # 스타일 손실
        style_loss = 0
        for gen_feat, style_feat in zip(generated_features[:-1], style_features):
            gen_gram = gram_matrix(gen_feat)
            style_gram = gram_matrix(style_feat)
            style_loss += style_weight * torch.mean((gen_gram - style_gram) ** 2)

        total_loss = content_loss + style_loss
        total_loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}, Total Loss: {total_loss.item()}")

    return generated_img

# 그램 행렬 계산
def gram_matrix(tensor):
    batch_size, channels, height, width = tensor.size()
    features = tensor.view(batch_size * channels, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * channels * height * width)

# 이미지 저장
def save_image(tensor, path):
    image = deprocess_image(tensor)
    output_image = Image.fromarray(image)
    output_image.save(path)

def main():
    # 이미지 경로 설정
    style_image_path = "uploads/styleImage/styleImage.png"
    content_image_path = "uploads/contentImage/contentImage.png"

    # 이미지 로드
    content_img = load_image(content_image_path)
    style_img = load_image(style_image_path)

    # 스타일 전이 수행
    result_img = style_transfer(content_img, style_img)

    # 결과 이미지 저장
    output_path = "output/result_image.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(result_img, output_path)
    print(f"Result image saved to {output_path}")

if __name__ == "__main__":
    main()
