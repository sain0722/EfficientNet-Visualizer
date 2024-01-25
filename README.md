# EfficientNet-Visualizer
EfficientNet 모델 결과를 Inference하는 프로그램입니다. Grad-CAM을 사용하여 시각화한 결과를 확인할 수 있습니다.

## Installation

---

**Requirements**

All the codes are tested in the following environment:
- Windows 10
- GPU: RTX 30xx
- Python 3.7
- torch 1.12.1
- torchvision 0.13.1

### Check Your CUDA Version
### GTX 1650 ~ RTX 2080

`CUDA` 10~10.2

`CUDNN` 7.5 ([Turing](https://en.wikipedia.org/wiki/Turing_(microarchitecture)))

### RTX 3050~3090

`CUDA` 11.1 ~ 11.4

`CUDNN` 8.6 ([Ampere](https://en.wikipedia.org/wiki/Ampere_(microarchitecture)))

### RTX 4060 ~ 4090 

`CUDA` 11.8 / 12.0~12.3

`CUDNN` 8.9 ([Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture)))

### 1. Install PyTorch
Install PyTorch for your GPU.

- Only CPU (Conda)
```
conda install pytorch==1.12.1 torchvision==0.13.1 -c pytorch
```

- Only CPU (pip)
```
pip install torch==1.12.1 torchvision==0.13.1
```

- GPU
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

### 2. Install EfficientNet-PyTorch
Install with `pip install efficientnet_pytorch` and load a pretrained EfficientNet with:
```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')
```

### 3. Install Grad-CAM for PyTorch
`pip install grad-cam`
```python
from pytorch_grad_cam import GradCAM
```

### 4. Install requirements
```
pip install -r requirements.txt
```

---

## Quick Demo
```
python main.py
```

![EfficientNetVisualizer_1.png](assets%2FEfficientNetVisualizer_1.png)
![EfficientNetVisualizer_2.png](assets%2FEfficientNetVisualizer_2.png)
![EfficientNetVisualizer_3.png](assets%2FEfficientNetVisualizer_3.png)
![EfficientNetVisualizer_4.png](assets%2FEfficientNetVisualizer_4.png)
