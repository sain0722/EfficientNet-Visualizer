import os
import shutil
import threading
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM, ScoreCAM, AblationCAM, XGradCAM, EigenCAM, FullGrad, \
    EigenGradCAM, RandomCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image


class DataLoaderWithPath(ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]  # 이미지의 파일 경로
        return img, label, path


def load_model(weight_path: str, num_classes: int = 1, model_number: int = 5):
    model_name = f'efficientnet-b{model_number}'
    model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    device_number = 0
    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")  # set gpu
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    model = model.to(device)

    return model


def load_example_model(model_number: int = 5):
    model_name = f'efficientnet-b{model_number}'
    model = EfficientNet.from_pretrained(model_name)
    device_number = 0
    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")  # set gpu
    model = model.to(device)
    model.eval()

    return model


def load_data(data_path: str, img_size: int = 456, sub_dir: bool = True):
    dataset = DataLoaderWithPath(
        root=data_path,
        transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    return dataset


def make_result_folder(name: str = "Default"):
    exp_root = './exp'
    if not os.path.exists(exp_root):
        os.makedirs(exp_root)

    test_root = f'{exp_root}/test'
    if not os.path.exists(test_root):
        os.makedirs(test_root)

    test_root = f'{test_root}/{name}'
    if not os.path.exists(test_root):
        os.makedirs(test_root)


def warm_up(model: EfficientNet, dataloader: DataLoader, device: torch.device):
    for i, (inputs, labels, path) in enumerate(dataloader):
        inputs = inputs.to(device)
        model(inputs)
        if i == 5:
            break


def heatmap(model, image, device, img_size, savename):
    # Grad-CAM에 사용할 모델의 타겟 레이어 설정
    target_layers = [model.extract_features]
    # 타겟 클래스 설정
    targets = [ClassifierOutputTarget(281)]  # 여기서 281을 변경할 수 있습니다.

    # image = Image.open(fname)
    # image = image.convert('RGB')
    raw_image = image

    rimage = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(image)
    images = torch.stack([rimage]).to(device)

    # Grad-CAM 객체 생성 및 실행
    # gcam = GradCAM(model=model, target_layers=target_layers, use_cuda=device.type == 'cuda')
    # grayscale_cam = gcam(input_tensor=images, targets=targets)
    gcam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = gcam(input_tensor=images, target_layer=targets)
    grayscale_cam = grayscale_cam[0, :]

    # 결과 저장
    plt.imshow(grayscale_cam, cmap='jet', alpha=0.5)
    plt.imshow(raw_image, alpha=0.5)
    plt.axis('off')
    plt.savefig(savename)
    plt.close()
    # visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)


def generate_gradcam(model, target_layer, device):
    # Grad-CAM 객체 생성
    # gcam = GradCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda')
    # gcam = EigenCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda')

    methods = {
        "GradCAM": GradCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        "GradCAM++": GradCAMPlusPlus(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        "EigenGradCAM": EigenGradCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        "AblationCAM": AblationCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        "RandomCAM": RandomCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        "HiResCAM": HiResCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        "ScoreCAM": ScoreCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        "XGradCAM": XGradCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        "EigenCAM": EigenCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        # "FullGrad": FullGrad(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda'),
    }

    gcam = methods["GradCAM"]
    # gcam = GradCAM(model=model, target_layers=[target_layer])
    return gcam


def generate_selected_cam(model, target_layer, device, method="GradCAM"):
    # Grad-CAM 객체 생성
    # gcam = GradCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda')
    # gcam = EigenCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda')

    methods = {
        "GradCAM": GradCAM(model=model,           target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        "GradCAM++": GradCAMPlusPlus(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        "EigenGradCAM": EigenGradCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        "AblationCAM": AblationCAM(model=model,   target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        "RandomCAM": RandomCAM(model=model,       target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        "HiResCAM": HiResCAM(model=model,         target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        "ScoreCAM": ScoreCAM(model=model,         target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        "XGradCAM": XGradCAM(model=model,         target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        "EigenCAM": EigenCAM(model=model,         target_layers=[target_layer], use_cuda=device.type == 'cuda'),
        # "FullGrad": FullGrad(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda'),
    }

    gcam = methods["GradCAM"]
    # gcam = GradCAM(model=model, target_layers=[target_layer])
    return gcam


def calculate_gradcam(gcam, input_tensor):
    targets = None  # 모델의 최종 예측 클래스에 대해 Grad-CAM을 계산
    grayscale_cam = gcam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    return grayscale_cam


def generate_cam_image(gcam, img, device, image_size):
    # 이미지 불러오기 및 전처리
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # Grad-CAM 계산
    st_time = time.time()
    targets = None  # 모델의 최종 예측 클래스에 대해 Grad-CAM을 계산
    grayscale_cam = gcam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    end_time = time.time()
    message = f"GradCAM 계산: {end_time - st_time:.3f} sec"
    print(message)

    # 원본 이미지를 numpy 배열로 변환
    rgb_image = input_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    rgb_image = (rgb_image * 255).astype(np.uint8)
    rgb_image = Image.fromarray(rgb_image)
    # rgb_image_resized = rgb_image.resize((456, 456))

    img_array = np.array(rgb_image) / 255

    # Grad-CAM을 원본 이미지 위에 표시
    cam_image = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

    # # 결과 시각화 및 저장
    # plt.imshow(cam_image)
    # plt.axis('off')
    # plt.savefig('gradcam_result.png')
    # plt.show()
    return cam_image


def main():
    name         = 'HYUNDAI_AR_PART1_E19_test_ng'
    nc           = 2
    model_number = 5
    img_size     = 456
    data_path    = r'D:\BIW\AI_Model\Hyundai_AR_Model\TRAIN_VALID_TEST_DATASET\1214_Hyundai_PART1\TEST/1.NG'
    weight_path  = r'D:\BIW\AI_Model\Hyundai_AR_Model\AI_MODEL\PART1\PART1.pt'
    class_list   = ["0.OK", "1.NG"]
    subdir       = True

    # 1. load_model
    model = load_model(weight_path, num_classes=nc, model_number=model_number)

    # 2. load_data
    test_dataset = load_data(data_path, img_size=img_size, sub_dir=subdir)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    if len(test_dataset) == 0:
        print("데이터셋 없음")
        return

    # 3. make_result_folder
    make_result_folder(name)

    # 4. warm_up
    device_number = 0
    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")  # set gpu
    warm_up(model, dataloader, device)

    # 5. Test and Save
    softmax = nn.Softmax(dim=1)

    exp_root = './exp'
    test_root = f'{exp_root}/test'
    test_root = f'{test_root}/{name}'

    model.eval()

    total = len(test_dataset)
    stime = time.time()
    with torch.no_grad():
        for i, (img, path) in enumerate(dataloader):
            fullname = path[0]
            fname = fullname.split('\\')[-1]

            # save_image(img[0], 'img1.png')

            print(img.shape)

            ## RGB to BGR
            # img = img[:,[2,1,0],:]

            img = img.to(device)

            outputs = model(img)
            if nc == 2:
                outputs = softmax(outputs)
            _, preds = torch.max(outputs, 1)

            outputs = outputs.cpu().detach().numpy()
            index = preds[0].cpu().numpy()
            p = class_list[index]
            print(f'[{i}/{total}]: {fname} => {p} {outputs[0][index] * 100}%')
            if not os.path.exists(f'{test_root}/{p}'):
                os.makedirs(f'{test_root}/{p}')

            # if index == 1:
            shutil.copyfile(fullname, f'{test_root}/{p}/{fname}')

    etime = time.time()
    ttime = etime - stime
    ttl_time = round(ttime, 2)
    avg_time = round(ttime / (i + 1) * 1000, 2)
    print(f"[time] total: {ttl_time}sec, avg: {avg_time}ms")


def grad_cam_main():
    name         = 'HYUNDAI_AR_PART1_E19_test_ng'
    nc           = 2
    model_number = 5
    img_size     = 456
    data_path    = r'D:\PROJECT\2024\BIW\data\efficient_net_data\240123_vuforia_456\test\검사항목#1\NG\cut_IMG_0746.PNG'
    weight_path  = r'D:\PROJECT\2024\BIW\240119\efficientNet\exp\20240123_target1_laptop_batch4_epoch49_acc1.0_batch_4.pt'
    class_list   = ["OK", "NG"]
    subdir       = True

    # 1. load_model
    model = load_model(weight_path, num_classes=nc, model_number=model_number)

    # 2. load_data
    # test_dataset = load_data(data_path, img_size=img_size, sub_dir=subdir)
    # dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    # if len(test_dataset) == 0:
    #     print("데이터셋 없음")
    #     return
    rgb_image = Image.open(data_path)
    rgb_image = rgb_image.convert('RGB')

    # 3. make_result_folder
    make_result_folder(name)

    # 4. warm_up
    device_number = 0
    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")  # set gpu
    # warm_up(model, dataloader, device)

    target_layer = model._conv_head  # EfficientNet의 마지막 컨볼루션 레이어
    gcam = generate_selected_cam(model, target_layer, device, method="GradCAM")
    model.eval()

    cam_image = generate_cam_image(gcam, rgb_image, device, img_size)
    cv2.imshow("cam_image", cam_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # main()
    grad_cam_main()
