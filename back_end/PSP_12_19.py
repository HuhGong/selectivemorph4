from pspNet import *
from torchvision.models import vgg19, VGG19_Weights
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch import optim
import numpy as np
from PIL import Image
import os

# 저장할 폴더 설정
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# 이미지 파일 경로 설정
style_image_file_path = "uploads/styleImage/styleImage.png"
content_image_file_path = "uploads/contentImage/contentImage.png"

# 이미지 로드
try:
    style_img = Image.open(style_image_file_path).convert("RGB")
    content_img = Image.open(content_image_file_path).convert("RGB")
except Exception as e:
    print(f"Error loading images: {e}")
    exit(1)

# custom_size: 그림 사이즈를 줄이는 변수로 설정할 예정
class_names = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor"
}
custom_size = (512, 512)
target_class_ids = [0]

# style_img 파일 로드
style_img = Image.open(style_image_file_path)
style_img_width, style_img_height = style_img.size
# plt.imshow(style_img)
# plt.show()

# content_image 파일 로드(segmentation 사진)
content_img = Image.open(content_image_file_path)
content_img_width, content_img_height = content_img.size
# plt.imshow(content_img)
# plt.show()


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

# 미리 학습 된 가중치로 vgg19모델 초기화하고
# fully connected layers를 제외한 특징 부분만 가져온다.(.features)
# 평가모드로 변화(.eval)
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

# 이미지 정규화 설정
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])


# transform: 이미지 전처리 수행
transform = DataTransform(input_size=475,
                          color_mean=cnn_normalization_mean,
                          color_std=cnn_normalization_std)
# 가중치 로드
state_dict = torch.load("D:\selectivemorph2\pspmodel\pspnet50_.pth", weights_only=True)

# 여러가지 이미지 리스트 생성.
rootpath = "D:\selectivemorph2\pspmodel\VOC2012"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

# PSPNet 모델을 21개의 클래스
# Pascal VOC 데이터셋: 이 데이터셋에는 20개의 객체 클래스와 1개의 배경 클래스를 포함하여 총 21개의 클래스
net = PSPNet(n_classes=21)
net.load_state_dict(state_dict) # 가중치 사용.
print("network configuration completed")


# 검증 이미지 및 어노테이션 로드
try:
    anno_file_path = val_anno_list[0]
    anno_class_img = Image.open(anno_file_path)
except Exception as e:
    print(f"Error loading annotation image: {e}")
    exit(1)

p_palette = anno_class_img.getpalette() # 팔레트 이미지의 색상 정보를 리스트로 반환.
phase = 'val'


img, anno_class_img = transform(phase, content_img, anno_class_img) # 학습에 적합한 상태로 이미지 전처리

# 평가 모드로 전환
net.eval()

x = img.unsqueeze(0) # c, h, w 형식의 텐서에 새로운 차원 추가.(배치)
outputs = net(x) # 세그먼테이션 모델 net 의 output은 (B, num_classes, H, W)
y = outputs[0] # output[0]은 (num_classes, H, W) 형식

y = y[0].detach().numpy()
# 첫 번째 클래스 맵을 선택한 것이 아니라, 모든 클래스 채널이 포함된 상태. (num_classes, H, W) 형식
# detach(기울기 추적 멈춤), numpy(넘파이 배열로 변환)
y = np.argmax(y, axis=0) # 클래스 예측 (H, W) 형식 각 픽셀에 대해 예측된 클래스 ID를 포함하는 2D 배열.


anno_class_img = Image.fromarray(np.uint8(y), mode="P") # PIL 이미지로 변환
anno_class_img = anno_class_img.resize((content_img_width, content_img_height), Image.NEAREST) # 이미지 사이즈 조정
anno_class_img.putpalette(p_palette) # 팔레트 값도 PIL 이미지에 적용. 팔레트는 각 클래스의 ID에 대응하는 RGB 색상 정의

# 어노테이션 이미지 저장
output_path = os.path.join(output_folder, "anno_class_img.png")
anno_class_img.save(output_path)


def print_class_colors(image, palette, class_names):
    """
    이미지와 팔레트를 기반으로 사용된 클래스 이름과 색상 정보를 출력합니다.
    :param image: PIL Image (mode="P") 또는 ndarray
    :param palette: 팔레트 정보 (리스트 형식, [R1, G1, B1, R2, G2, B2, ...])
    :param class_names: 클래스 ID와 이름 매핑 (딕셔너리)
    """
    # 클래스 ID 가져오기
    image_array = np.array(image)  # 이미지 배열로 변환
    unique_classes = np.unique(image_array)  # 사용된 클래스 ID 확인

    print("\n사용된 클래스와 색상 정보:")
    for class_id in unique_classes:
        # 클래스 이름 가져오기
        class_name = class_names.get(class_id, "Unknown")  # ID가 없으면 Unknown
        # 팔레트에서 클래스 색상 추출 (ID에 따라 RGB 값 가져오기)
        color = palette[class_id * 3: class_id * 3 + 3]  # RGB 값 가져오기
        print(f"Class ID {class_name} (ID: {class_id}): Color {tuple(color)}")

# 사용된 클래스 이름과 색상 출력 convert("P") 는 팔레트로 변환을 의미한다.
print_class_colors(anno_class_img.convert("P"), p_palette, class_names)

def create_class_mask(segmentation_map, target_class_ids):
    """
    특정 클래스 ID에 대한 마스크를 생성합니다.
    """
    segmentation_array = np.array(segmentation_map)
    mask = np.isin(segmentation_array, target_class_ids).astype(np.uint8)  # 특정 클래스만 1로 설정
    return mask

resized_anno_class_img = resize_with_aspect_ratio_preserve_ids(anno_class_img, target_size=custom_size)
# plt.imshow(resized_anno_class_img)
# plt.show()
segmentation_map = resized_anno_class_img.convert("P")  # 세그멘테이션 결과로 얻은 클래스 맵 이미지.
segmentation_array = np.array(segmentation_map)  # 세그멘테이션 맵 배열
unique_classes_in_map = np.unique(segmentation_array)  # 고유 클래스 ID 추출
print("Classes in Segmentation Map:", unique_classes_in_map)

class_mask = create_class_mask(segmentation_map, target_class_ids)
# plt.imshow(class_mask, cmap='gray')
# plt.show()


# 마스크를 원본 이미지에 적용
resized_content_img = resize_with_aspect_ratio(content_img, target_size=custom_size)
# plt.imshow(resized_anno_class_img)
# plt.show()
content_image_array = np.array(resized_content_img.convert("RGBA"))  # 원본 이미지 RGBA
class_mask_expanded = np.stack([class_mask] * 4, axis=-1)  # class_mask이미지도 원본 처럼 동일하게 RGBA로 확장해야 호환이 된다.

# 특정 클래스만 추출
masked_content = content_image_array * class_mask_expanded # 둘이 곱하면 특정 class 영역만 유지. 마스크 값이 1인 픽셀만 원본 이미지 유지
masked_content_image = Image.fromarray(masked_content)  # PIL 이미지로 변환
masked_content_path = os.path.join(output_folder, "masked_content.png")
masked_content_image.save(masked_content_path)

# 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# 이미지 로더 설정
# gpu 사용시 512, cpu 사용시 128(성능 이슈 때문)
imsize = 512 if torch.cuda.is_available() else 128
loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()]) # 이미지 사이즈 조정 후 텐서로 변경.

# image_loader: 학습 또는 추론이 가능하게 텐서 변환을 수행하는 함수
def image_loader(image, resize=None):
    # Open the image
    image = image.convert('RGB')

    # Resize the image if a resize value is provided
    if resize:
        transform = transforms.Compose([
            transforms.Resize(resize),  # Resize to specified dimensions
            transforms.ToTensor(),  # 텐서로 변환과 동시에 [0,255] -> [0,1]로 정규화, 출력 텐서 크기: (C, H, W)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor
        ])

    # Apply transformations and add batch dimension
    image = transform(image).unsqueeze(0) # 파이토치는(B, C, H, W) 형식의 입력을 요구 즉 차원 추가

    # Move to specified device
    return image.to(device, torch.float) # torch.float: 텐서 타입을 float 32로 변환

# 위 코드에서는 2가지 (anno_class_img, masked_content) 가 핵심.
###################################################################################################
# 여기서 부터는 style transfer


# 마스크가 적용된 이미지(내가 원하는 class만 추출한 이미지)를 image_loader 함수로 변환 또는 이동
content_img_resized = resize_with_aspect_ratio(masked_content_image, target_size=custom_size)
# plt.imshow(content_img_resized)
# plt.show()

# 스타일 트랜스퍼 준비
content_image = image_loader(content_img_resized)

# Get the size of the content image
content_size = content_image.shape[2:]  # (C, H, W) 중에서 H, W만 가져온다.

# 스타일을 적용할 meta 이미지를 마찬가지로 image_loader로 변환하고 사이즈도 content_image로 맞춤.
style_img_resized = resize_with_aspect_ratio(style_img, target_size=custom_size)
# plt.imshow(style_img_resized)
# plt.show()
style_image = image_loader(style_img_resized)

assert style_image.size() == content_image.size(), "we need to import style and content images if the same size" # style_image와 content_image가 사이즈가 동일한지 확인

unloader = transforms.ToPILImage()  # 텐서를 다시 PIL로 변환(시각화 하기 위해 필요)

def imshow(tensor, title=None):
    image = tensor.cpu().clone() # 원본은 건드리지 않고 복사하여 cpu로 이동
    image = image.squeeze(0) # 배치 제거하고 시각화에 필요한 (C, H, W) 형태로 변환
    image = unloader(image) # 위에 정의한 uploader로 PIL로 변환
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # 0.001초 동안 일시정지하며 그래프 업데이트


input_img = content_image.clone()


# 입력 이미지가 content 이미지의 특징을 얼마나 잘 보존하는지 측정하는 class
class ContentLoss(nn.Module):
    # target = content 이미지에서 추출된 feature 텐서
    def __init__(self, target):
        super().__init__()
        self.target = target.detach() # target은 back propagation 시에 변경되지 않게 detach
        # content 이미지의 특징은 고정된 값으로 손실 값의 기준
    def forward(self, input_):
        self.loss = F.mse_loss(input_, self.target) # 손실 계산
        return input_ # 인렵 텐서 그대로 반환하여 forward propagation 흐름 유지

# 스타일 트랜스퍼에서 스타일 이미지의 스타일 특성을 표현, feature map간의 내적 관계를 나타냄
def gram_matrix(input_):
    a, b, c, d = input_.size()

    features = input_.view(a * b, c * d) # 특징맵을 2d로 flatten(a: 배치, b: 채널 수, c, d: 특징 맵 높이 너비)
    G = torch.mm(features, features.t()) # features와 전치행렬 features 간의 곱

    return G.div(a * b * c * d) # 정규화를 (a * b * c * d)로 나누어서 한다.


# 스타일 이미지와 생성된 이미지의 gram matrix 차이를 이용해 styleloss 계산
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input_):
        G = gram_matrix(input_)
        self.loss = F.mse_loss(G, self.target)
        return input_


# vgg 모델을 불러와 특징 추출용으로 사용.
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])


# 정규화 클래스에서 정규화 된 이미지는 vgg19모델에 입력되고 이 모델의 스타일 손실과 콘텐츠 손실을 계산.
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # 평균과 표준편차에 파이토치 텐서로 변환하고 형태 맞춤
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    # 정규화 공식 사용해 이미지 정규화
    def forward(self, img):
        return (img - self.mean) / self.std


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


# style transfer 작업에서 vgg19 네트워크를 사용해 Content Loss와 Style Loss를 계산할 수 있도록 준비
# 파이토치로 새로운 모델 생성하고 content loss와 style loss를 삽입하는 과정을 구현
def get_style_model_and_losses(cnn, normalization_mean,
                               normalization_std,
                               style_img, content_img, content_layers=content_layers_default,
                               style_layers=style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)

            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses



def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])
    return optimizer




# content image와 style image를 기반으로 입력 이미지를 최적화해 스타일 트랜스퍼 결과 이미지를 생성하는 함수
# input_img: 최적화를 시작할 입력 이미지 (일반적으로 content_img의 복사본).
# num_steps: 최적화 반복 횟수 (기본값: 300).
# style_weight: 스타일 손실의 가중치.
# content_weight: 콘텐츠 손실의 가중치.
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    print("Building model...")
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean,
                                                                     normalization_std,
                                                                     style_img, content_img)
    input_img.requires_grad_(True) # 인풋 이미지 최적화 대상으로 설정

    model.eval() # 이 코드는 평가 모드로 설정하여 drop out 같은 훈련중 활성되는 요소를 비활성한다.
    model.requires_grad_(False) # 모델 가중치 고정

    optimizer = get_input_optimizer(input_img)

    print("Optimizing...")
    run = [0] # 최적화 횟수를 저장하는 리스트
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1) # 인풋이미지의 값을 0~1로 클리핑

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            # 손실값 누적
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            # 손실 가중치 적용 및 최적화
            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print("Style Loss: {:.4f}".format(style_score.item()))
                print("Content Loss: {:.4f}".format(content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


output = run_style_transfer(cnn, cnn_normalization_mean,
                            cnn_normalization_std,
                            content_image,
                            style_image, input_img)

output_image_path = os.path.join(output_folder, 'imgtrans.png')
save_image(output, output_image_path)

# 3. 스타일 트랜스퍼 결과와 원본 이미지 합성
# 스타일 트랜스퍼 결과 이미지
style_transferred_img = Image.open(output_image_path).convert("RGBA")
style_transferred_array = np.array(style_transferred_img)


# 원본 이미지
original_image = Image.open(content_image_file_path).convert("RGBA")
resized_original_image = resize_with_aspect_ratio(original_image, target_size=custom_size)
original_array = np.array(resized_original_image)

# 스타일 트랜스퍼된 클래스 영역만 적용(class_mask_expanded: 특정 클래스 영역을 나타내는 0 또는 1로 구성된 마스크 배열)
# np.where(condition, x, y) (처음 보는 함수....)
# class_mask_expanded = (1이면 스타일 트랜스퍼 이미지 사용, 0이면 원본이미지 사용)
result_array = np.where(class_mask_expanded, style_transferred_array, original_array)

# 합성된 이미지 저장
final_result = Image.fromarray(result_array.astype(np.uint8)) # 다시 PIL로 변경
final_result_path = os.path.join(output_folder, "final_combined_image.png")
final_result.save(final_result_path)

print(f"최종 결과 이미지가 '{final_result_path}'에 저장되었습니다.")





