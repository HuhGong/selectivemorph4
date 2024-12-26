import torch
from torchvision import models, transforms
from torchvision.utils import save_image
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_vgg19_model():
    """
    VGG19 모델을 로드하여 특징 추출을 위한 준비를 합니다.
    """
    vgg = models.vgg19(weights='DEFAULT').features.eval().to(device)
    return vgg

def preprocess_image(image):
    """
    이미지를 VGG19 모델 입력에 맞게 전처리합니다.
    """
    transform = transforms.Compose([
        transforms.Resize((720, 720)),  # 입력 크기에 맞게 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device)  # 배치 차원 추가 및 장치로 이동

def deprocess_image(tensor):
    """
    텐서를 이미지로 변환합니다.
    """
    tensor = tensor.to("cpu").clone().detach()
    tensor = tensor.squeeze(0)  # 배치 차원 제거
    tensor = (tensor * 0.5) + 0.5  # 정규화된 값을 다시 [0, 1]로 변환
    tensor = tensor.clamp(0, 1)  # [0, 1] 범위로 클램프
    return transforms.ToPILImage()(tensor)

def style_transfer(content_image, style_image, num_steps=300, style_weight=1000000, content_weight=1):
    """
    스타일 전송을 수행하는 함수입니다.
    """
    vgg = load_vgg19_model()

    # 이미지 전처리
    content_tensor = preprocess_image(content_image)
    style_tensor = preprocess_image(style_image)

    # 입력 이미지 초기화
    input_tensor = content_tensor.clone()

    # 손실을 추적하기 위한 레이어
    content_layers = ['21']  # VGG19의 content layer
    style_layers = ['0', '5', '10', '19', '28']  # VGG19의 style layers

    # 손실 저장을 위한 딕셔너리
    content_losses = []
    style_losses = []

    # VGG19의 레이어를 순회하며 손실을 계산
    for layer in vgg.children():
        input_tensor = layer(input_tensor)
        if str(layer) in content_layers:
            target = layer(content_tensor)
            content_loss = torch.nn.functional.mse_loss(input_tensor, target)
            content_losses.append(content_loss * content_weight)

        if str(layer) in style_layers:
            target = layer(style_tensor)
            style_loss = torch.nn.functional.mse_loss(input_tensor, target)
            style_losses.append(style_loss * style_weight)

    # 총 손실 계산
    total_loss = sum(content_losses) + sum(style_losses)

    # 최적화
    optimizer = torch.optim.LBFGS([input_tensor.requires_grad_()])

    for i in range(num_steps):
        def closure():
            optimizer.zero_grad()
            # 손실 재계산
            total_loss.backward()
            return total_loss

        optimizer.step(closure)

    # 결과 이미지 반환
    return deprocess_image(input_tensor)

def transfer_selected_class(selected_class_image, style_image, num_steps=300, style_weight=1000000, content_weight=1):
    """
    선택한 클래스 이미지를 스타일 전송하는 함수입니다.
    """
    return style_transfer(selected_class_image, style_image, num_steps, style_weight, content_weight)

if __name__ == "__main__":
    # 예시 이미지 로드
    content_image_path = "uploads/contentImage/contentImage.png"  # 콘텐츠 이미지 경로
    style_image_path = "uploads/styleImage/styleImage.png"  # 스타일 이미지 경로

    content_image = Image.open(content_image_path).convert("RGB")
    style_image = Image.open(style_image_path).convert("RGB")

    # 스타일 전송 수행
    result_image = transfer_selected_class(content_image, style_image)

    # 결과 이미지 저장
    result_image.save("output/result_image.png")
    print("스타일 전송 완료, 결과 이미지가 'output/result_image.png'에 저장되었습니다.")
