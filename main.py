import json
import sys
import os
from datetime import datetime

import PyQt5
import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QPushButton, \
    QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, \
    QMainWindow, QComboBox, QListWidget, QListWidgetItem, QHBoxLayout, QCheckBox
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
from utils import load_model, load_data, warm_up
import torch
import torch.nn as nn

from PIL import Image, ImageDraw, ImageFont


class EfficientNetVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = None
        self.test_dataset = None
        self.dataloader = None
        self.is_example_model = False

        self.resize(1800, 1000)
        self.setWindowTitle("EfficientNetVisualizer")

        self.central_widget = QWidget(self)
        self.hlayout_main = QtWidgets.QHBoxLayout(self.central_widget)
        self.hlayout_main.setObjectName("MainLayout")
        self.hlayout_main.setSpacing(10)
        self.vlayout_main = QtWidgets.QVBoxLayout()

        # Widget Definition
        self.widget_input = QtWidgets.QWidget(self.central_widget)

        self.vlayout_input = QtWidgets.QVBoxLayout(self.widget_input)
        self.vlayout_input.setObjectName("vlayout_input")

        # 사용자 입력부
        self.lbl_model_name = QtWidgets.QLabel("테스트 모델명", self.widget_input)
        self.line_edit_model_name = QtWidgets.QLineEdit("EfficientNet_Test", self.widget_input)

        self.lbl_num_class = QtWidgets.QLabel("클래스 개수", self.widget_input)
        self.sbx_num_class = QtWidgets.QSpinBox(self.widget_input)
        self.sbx_num_class.setMinimum(1)
        self.sbx_num_class.setMaximum(1000)
        self.sbx_num_class.setValue(2)

        self.lbl_class_name = QtWidgets.QLabel("클래스 명(콤마(,)로 구분)", self.widget_input)
        self.line_edit_class_name = QtWidgets.QLineEdit("NG, OK", self.widget_input)
        self.btn_class_name = QPushButton("클래스 명 적용", self.widget_input)
        self.btn_class_name.clicked.connect(self.on_submit_class_name)

        self.lbl_img_size = QtWidgets.QLabel("이미지 사이즈", self.widget_input)
        self.cbx_img_size = QComboBox(self.widget_input)
        # 콤보 박스 항목 추가
        sizes = ["224", "240", "260", "300", "380", "456", "528", "600"]
        for size in sizes:
            self.cbx_img_size.addItem(size)
        self.cbx_img_size.currentIndexChanged.connect(self.on_image_size_changed)

        self.lbl_model_number = QtWidgets.QLabel("EfficientNet 모델 번호", self.widget_input)
        self.line_edit_model_number = QtWidgets.QLineEdit("5", self.widget_input)
        self.line_edit_model_number.setEnabled(False)
        self.cbx_img_size.setCurrentIndex(5)

        self.btn_select_data = QPushButton("데이터 선택(Directory)", self.widget_input)
        self.lbl_data_path = QtWidgets.QLabel("선택된 데이터", self.widget_input)
        self.lbl_data_path_value = QtWidgets.QLineEdit("{경로}", self.widget_input)
        self.lbl_data_path_value.setReadOnly(True)

        self.btn_select_weight = QPushButton("가중치(Weight) 선택", self.widget_input)
        self.lbl_weight_path = QtWidgets.QLabel("선택된 가중치", self.widget_input)
        self.lbl_weight_path_value = QtWidgets.QLineEdit("{경로}", self.widget_input)
        self.lbl_weight_path_value.setReadOnly(True)

        # Layout에 추가
        self.vlayout_input.addWidget(self.lbl_model_name)
        self.vlayout_input.addWidget(self.line_edit_model_name)

        self.vlayout_input.addWidget(self.lbl_num_class)
        self.vlayout_input.addWidget(self.sbx_num_class)

        self.vlayout_input.addWidget(self.lbl_class_name)
        self.vlayout_input.addWidget(self.line_edit_class_name)
        self.vlayout_input.addWidget(self.btn_class_name)

        self.vlayout_input.addWidget(self.lbl_img_size)
        self.vlayout_input.addWidget(self.cbx_img_size)
        self.vlayout_input.addWidget(self.lbl_model_number)
        self.vlayout_input.addWidget(self.line_edit_model_number)

        self.vlayout_input.addWidget(self.btn_select_data)
        self.vlayout_input.addWidget(self.lbl_data_path)
        self.vlayout_input.addWidget(self.lbl_data_path_value)

        self.vlayout_input.addWidget(self.btn_select_weight)
        self.vlayout_input.addWidget(self.lbl_weight_path)
        self.vlayout_input.addWidget(self.lbl_weight_path_value)

        # EfficientNet 관련 기능 선언
        self.btn_load_example_model = QPushButton("테스트 모델 Load (테스트용)", self.widget_input)
        self.combobox_test_data = QComboBox(self.widget_input)
        self.btn_test_model_run = QPushButton("테스트 모델 결과 확인 (테스트용)", self.widget_input)
        self.btn_load_model = QPushButton("모델 Load", self.widget_input)
        self.btn_load_data = QPushButton("데이터 Load", self.widget_input)
        self.btn_test_run = QPushButton("모델 결과 확인", self.widget_input)
        self.cbx_gradcam = QCheckBox("시각화", self.widget_input)

        self.vlayout_input.addWidget(self.btn_load_model)
        self.vlayout_input.addWidget(self.btn_load_data)
        self.vlayout_input.addWidget(self.btn_test_run)
        self.vlayout_input.addWidget(self.btn_load_example_model)
        self.vlayout_input.addWidget(self.combobox_test_data)
        self.vlayout_input.addWidget(self.btn_test_model_run)
        self.vlayout_input.addWidget(self.cbx_gradcam)

        # Widget Log
        self.widget_log = QtWidgets.QListWidget(self.widget_input)
        self.widget_log.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.widget_log.setFlow(QtWidgets.QListView.TopToBottom)
        self.widget_log.setObjectName("widget_log")
        self.vlayout_input.addWidget(self.widget_log)

        # spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        # self.vlayout_input.addItem(spacerItem)

        # Widget Image
        self.hlayout_main.addWidget(self.widget_input)

        # Test Widget (이미지 슬라이더)
        self.widget_image_slider = ImageDisplayWidget()
        self.hlayout_main.addWidget(self.widget_image_slider)

        self.hlayout_main.setStretch(0, 3)
        self.hlayout_main.setStretch(1, 7)

        # self.setLayout(self.hlayout_main)
        self.setCentralWidget(self.central_widget)

        self.init_signals()
        self.initialize()

    def init_signals(self):
        self.btn_select_data.clicked.connect(self.select_test_data_directory)
        self.btn_select_weight.clicked.connect(self.select_weight_path)
        self.btn_load_example_model.clicked.connect(self.load_example_model)
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_load_data.clicked.connect(self.load_data)
        # self.btn_test_run.clicked.connect(self.test_run)
        self.btn_test_run.clicked.connect(self.test_run_thread)
        self.btn_test_model_run.clicked.connect(self.test_model_run)

    def on_image_size_changed(self, index):
        self.line_edit_model_number.setText(str(index))

    def initialize(self):
        data_path = r"D:\PROJECT\2024\BIW\data\efficient_net_data\240123_vuforia_456\test\검사항목#1"
        weight_path = r"D:\PROJECT\2024\BIW\240119\efficientNet\exp\20240123_target1_laptop_batch4_epoch49_acc1.0_batch_4.pt"
        self.lbl_data_path_value.setText(data_path)
        self.lbl_weight_path_value.setText(weight_path)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        test_image_path = "src"
        try:
            test_images = [os.path.join(test_image_path, filename)
                           for filename in os.listdir(test_image_path)
                           if any(filename.lower().endswith(ext) for ext in image_extensions)]
            self.combobox_test_data.addItems(test_images)
        except FileNotFoundError:
            self.write_log("테스트 데이터가 담긴 src 폴더를 찾을 수 없습니다.")

    def write_log(self, message):
        result = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        self.widget_log.addItem(result)
        current_count = self.widget_log.count()
        if current_count >= 200:
            # 최대 항목 수를 초과하는 경우 가장 오래된 항목을 제거
            self.widget_log.takeItem(0)

        self.widget_log.scrollToBottom()

    def on_submit_class_name(self):
        text = self.line_edit_class_name.text()
        try:
            input_list = [str(item.strip()) for item in text.split(',')]
            num_classes = self.sbx_num_class.value()

            if len(input_list) != num_classes:
                message = f"[Error] 설정된 클래스 개수와 입력된 클래스 이름의 개수가 일치하지 않습니다."
            else:
                message = f"[Complete] Entered list: {input_list}"

            self.write_log(message)

        except ValueError:
            message = "Invalid input. Please enter a list of numbers separated by commas."
            self.write_log(message)

    def select_test_data_directory(self):
        data_dir_path = QFileDialog.getExistingDirectory(None, 'Select Directory')
        if data_dir_path:
            self.lbl_data_path_value.setText(data_dir_path)

    def select_data_path(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("Image Files (*.png *.jpg *.bmp)")
        if dialog.exec_():
            selected_files = dialog.selectedFiles()[0]
            if selected_files:
                self.lbl_data_path_value.setText(selected_files)

    def select_weight_path(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("Weight Files (*.pt *.pth)")
        if dialog.exec_():
            selected_files = dialog.selectedFiles()[0]
            if selected_files:
                self.lbl_weight_path_value.setText(selected_files)

    def load_example_model(self):
        model_number = int(self.line_edit_model_number.text())
        try:
            self.model = utils.load_example_model(model_number)
            self.is_example_model = True
            self.write_log("테스트 모델 로드 완료")
        except RuntimeError as e:
            print(e)
            QMessageBox.critical(self, "실패", f"모델 로드 실패. 에러 로그를 확인하세요.")
            self.write_log(e)
            return

    def load_model(self):
        weight_path = self.lbl_weight_path_value.text()
        num_classes = self.sbx_num_class.value()
        model_number = int(self.line_edit_model_number.text())
        try:
            self.model = load_model(weight_path, num_classes, model_number)
        except RuntimeError as e:
            print(e)
            QMessageBox.critical(self, "실패", f"모델 로드 실패. 에러 로그를 확인하세요.")
            self.write_log(e)
            return
        except FileNotFoundError:
            QMessageBox.critical(self, "실패", f"모델 로드 실패. 설정된 모델(가중치) 위치를 확인하세요.")
            return

        self.write_log("모델 로드 성공")
        self.is_example_model = False

    def load_data(self):
        data_path = self.lbl_data_path_value.text()
        img_size = int(self.cbx_img_size.currentText())
        sub_dir = True
        try:
            self.test_dataset = load_data(data_path, img_size, sub_dir)
        except FileNotFoundError:
            QMessageBox.critical(self, "실패", f"Couldn't find any class folder in {data_path}.")
            return

        self.dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0)

        if len(self.test_dataset) == 0:
            self.write_log("데이터 로드 실패")
            return

        device_number = 0
        device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")  # set gpu
        warm_up(self.model, self.dataloader, device)

        self.write_log("데이터 로드 성공")

    def test_run_thread(self):
        if self.model is None:
            QMessageBox.critical(self, "실패", "모델이 등록되지 않았습니다.")
            return

        if self.dataloader is None:
            QMessageBox.critical(self, "실패", "데이터가 등록되지 않았습니다.")
            return

        num_class = self.sbx_num_class.value()
        text = self.line_edit_class_name.text()
        class_list = [str(item.strip()) for item in text.split(',')]

        device_number = 0
        self.device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")  # set gpu

        # UI (리스트 초기화)
        self.widget_image_slider.imageListWidget.clear()

        # GradCAM
        target_layer = self.model._conv_head  # EfficientNet의 마지막 컨볼루션 레이어
        self.gcam = utils.generate_gradcam(self.model, target_layer, self.device)
        self.model.eval()

        self.ui_update_worker = UiUpdateWorker(self, self.gcam)
        self.inference_worker = InferenceWorker(self.model, self.dataloader, self.device, num_class, class_list,
                                                self.ui_update_worker)

        self.inference_worker.finished.connect(self.write_log)
        self.inference_worker.start()

    def test_model_run(self):
        if not self.combobox_test_data.currentText():
            QMessageBox.critical(None, "경고", "테스트 데이터가 설정되지 않았습니다.")
            return

        # Load class names
        try:
            num_class = 1000
            labels_map = json.load(open('src/labels_map.txt'))
            labels_map = [labels_map[str(i)] for i in range(num_class)]
        except FileNotFoundError:
            QMessageBox.critical(None, "경고", "labels_map.txt 파일을 찾을 수 없습니다.")
            return

        image_size = int(self.cbx_img_size.currentText())
        filename = self.combobox_test_data.currentText()
        pil_image = Image.open(filename)
        # Preprocess image
        tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])
        img = tfms(pil_image).unsqueeze(0)

        rgb_image = img.cpu().squeeze().permute(1, 2, 0).numpy()
        rgb_image = (rgb_image * 255).astype(np.uint8)

        device_number = 0
        self.device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")  # set gpu

        with torch.no_grad():
            inputs = img.to(self.device)
            logits = self.model(inputs)
        preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

        print('-----')
        log = ""
        for idx in preds:
            label = labels_map[idx]
            prob = torch.softmax(logits, dim=1)[0, idx].item()
            print('{:<75} ({:.2f}%)'.format(label, prob * 100))
            text = '{:<75} ({:.2f}%)\n'.format(label, prob * 100)
            log += text

        # GradCAM
        device_number = 0
        self.device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")  # set gpu

        target_layer = self.model._conv_head  # EfficientNet의 마지막 컨볼루션 레이어
        self.gcam = utils.generate_gradcam(self.model, target_layer, self.device)
        self.model.eval()

        self.ui_update_worker = UiUpdateWorker(self, self.gcam)

        index = preds[0]
        pred_class = labels_map[index]
        pred_class = pred_class.split(",")[0]
        score = torch.softmax(logits, dim=1)[0, index].item()
        prediction_text = f'{pred_class} {score * 100:.3f}%'
        log = f'[Test]\n{log}'
        print(log)

        # self.progress.emit(rgb_image, log, prediction_text, i)
        self.ui_update_worker.update_param(rgb_image, log, prediction_text, filename)
        self.ui_update_worker.start()
        self.ui_update_worker.wait()


class GraphicView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.pixmap_item = None

    def setScenePixmap(self, scene: QGraphicsScene, pixmap_item: QGraphicsPixmapItem) -> None:
        self.setScene(scene)
        self.pixmap_item = pixmap_item

    def wheelEvent(self, event):
        # 마우스 휠 이벤트 처리
        delta = event.angleDelta().y()
        if delta > 0:
            self.scale(1.1, 1.1)  # 확대
        else:
            self.scale(0.9, 0.9)  # 축소


class ImageDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.layout = QHBoxLayout(self)

        self.imageListWidget = QListWidget(self)
        # self.imageListWidget.itemClicked.connect(self.onImageSelected)
        self.imageListWidget.currentItemChanged.connect(self.onImageSelected)
        self.gview_image = GraphicView()  # GraphicView 사용
        self.layout.addWidget(self.gview_image)
        self.layout.addWidget(self.imageListWidget)
        self.layout.setStretch(0, 10)
        self.layout.setStretch(1, 1)

    def addImageToList(self, pixmap, name):
        item = QListWidgetItem(name)
        self.imageListWidget.addItem(item)
        item.setData(Qt.UserRole, pixmap)

    def onImageSelected(self, item):
        if item is None:
            print("item is None")
            return

        pixmap = item.data(Qt.UserRole)
        pixmap_item = QGraphicsPixmapItem(pixmap)  # QGraphicsPixmapItem 생성
        self.scene = QGraphicsScene()
        self.scene.addItem(pixmap_item)
        self.gview_image.setScenePixmap(self.scene, pixmap_item)
        self.gview_image.fitInView(pixmap_item, Qt.KeepAspectRatio)


class InferenceWorker(QThread):
    progress = PyQt5.QtCore.pyqtSignal(np.ndarray, str, str, int)
    finished = PyQt5.QtCore.pyqtSignal(str)

    def __init__(self, model, dataloader, device, num_class, class_list, ui_update_worker):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.num_class = num_class
        self.class_list = class_list

        self.ui_update_worker = ui_update_worker

    def run(self):
        correct_predictions = 0
        total_predictions = 0
        total_score = 0.0  # 전체 점수 누적 변수

        with torch.no_grad():
            for i, (inputs, labels, paths) in enumerate(self.dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                filename = os.path.basename(paths[0])

                rgb_image = inputs.cpu().squeeze().permute(1, 2, 0).numpy()
                rgb_image = (rgb_image * 255).astype(np.uint8)

                outputs = self.model(inputs)
                if self.num_class == 2:
                    softmax = nn.Softmax(dim=1)
                    outputs = softmax(outputs)
                _, preds = torch.max(outputs, 1)

                # 실제 레이블과 예측 결과 비교
                correct_predictions += torch.sum(preds == labels.data).item()
                total_predictions += inputs.size(0)
                total_score += outputs[0][preds[0]].item()  # 예측된 클래스의 확률 점수 추가

                outputs = outputs.cpu().detach().numpy()
                index = preds[0].cpu().numpy()
                pred_class = self.class_list[index]
                prediction_text = f'{pred_class} {outputs[0][index] * 100:.3f}%'
                log = f'[{i + 1}/{len(self.dataloader)}]: {filename} => {prediction_text}'
                print(log)

                # self.progress.emit(rgb_image, log, prediction_text, i)
                self.ui_update_worker.update_param(rgb_image, log, prediction_text, filename)
                self.ui_update_worker.start()
                self.ui_update_worker.wait()

        accuracy = correct_predictions / total_predictions
        average_score = total_score / total_predictions  # 평균 점수 계산
        print(f'Overall Accuracy: {accuracy:.2f}, Average Score: {average_score:.4f}')
        result_log = f'Overall Accuracy: {accuracy:.2f}, Average Score: {average_score:.4f}'
        self.finished.emit(result_log)


class UiUpdateWorker(QThread):
    finished = PyQt5.QtCore.pyqtSignal()

    def __init__(self, parent: EfficientNetVisualizer, gcam):
        super().__init__()
        self.parent = parent
        self.gcam = gcam

        self.rgb_image = None
        self.log = None
        self.prediction_text = None
        self.filename = None

    def update_param(self, rgb_image, log, prediction_text, filename):
        self.rgb_image = rgb_image
        self.log = log
        self.prediction_text = prediction_text
        self.filename = filename

    def run(self):
        self.parent.write_log(self.log)
        rgb_image = Image.fromarray(self.rgb_image)
        with torch.enable_grad():
            if self.parent.cbx_gradcam.isChecked():
                # st_time = time.time()
                image_size = int(self.parent.cbx_img_size.currentText())
                cam_image = utils.generate_cam_image(self.gcam, rgb_image, self.parent.device, image_size)
                display_image = Image.fromarray(cam_image)
                # end_time = time.time()
                # message = f"GradCAM : {end_time - st_time:.3f} sec"
                # self.write_log(message)
            else:
                display_image = rgb_image

        img = display_rgb_image_with_text(display_image, self.prediction_text)

        data_path = self.parent.lbl_data_path_value.text()
        file_name = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S%f')}.jpg"
        save_path = os.path.join(data_path, "../", "Test_Inference")
        os.makedirs(save_path, exist_ok=True)
        img.save(os.path.join(save_path, file_name))

        data = img.tobytes()
        qim = QImage(data, img.size[0], img.size[1], QImage.Format_RGB888)
        pixmap_image = QPixmap.fromImage(qim)

        pixmap_item = QGraphicsPixmapItem(pixmap_image)
        scene = QGraphicsScene()
        scene.addItem(pixmap_item)
        view = self.parent.widget_image_slider.gview_image
        view.setScenePixmap(scene, pixmap_item)
        view.fitInView(pixmap_item, Qt.KeepAspectRatio)

        self.parent.widget_image_slider.addImageToList(pixmap_image, self.filename)
        self.finished.emit()


def get_qimage(image):
    if image.shape[-1] == 3:
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        image_format = QImage.Format_RGB888

        # composing image from image data
        image = QImage(bytes(image.data),
                       width,
                       height,
                       bytesPerLine,
                       image_format)

        image = image.rgbSwapped()

    else:
        # 데이터 타입을 uint8로 변환합니다.
        if image.max() != 0:
            depth_data_uint8 = cv2.convertScaleAbs(image, alpha=(255.0 / image.max()))
        else:
            depth_data_uint8 = image

        # Convert the image to grayscale
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = depth_data_uint8.shape
        bytesPerLine = width
        image_format = QImage.Format_Grayscale8

        # composing image from image data
        image = QImage(depth_data_uint8.data,
                       width,
                       height,
                       bytesPerLine,
                       image_format)

    return image


def set_graphic_view_image(image, view: GraphicView):
    qimage = get_qimage(image)
    pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(qimage))
    scene = QGraphicsScene()
    scene.addItem(pixmap_item)
    view.setScenePixmap(scene, pixmap_item)
    view.fitInView(pixmap_item, Qt.KeepAspectRatio)


def display_tensor_image_with_text(img_tensor, text):
    # 이미지 변환: PyTorch Tensor -> PIL Image
    img = img_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)

    # PIL을 사용하여 이미지에 텍스트 추가
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 30)
    text_width, text_height = draw.textsize(text, font=font)
    xy = (0, 0), (text_width, text_height)
    draw.rectangle(xy, fill="black")
    draw.text((0, 0), text, font=font, fill="white")

    # PyQt에서 사용할 수 있도록 QPixmap 객체로 변환
    img = img.convert("RGB")
    data = img.tobytes()
    qim = QImage(data, img.size[0], img.size[1], QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qim)

    return pixmap


def display_rgb_image_with_text(img, text):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 30)

    # 텍스트의 경계 상자 계산
    text_bbox = draw.textbbox((0, 0), text, font=font)
    # text_width = text_bbox[2] - text_bbox[0]
    # text_height = text_bbox[3] - text_bbox[1]

    # 텍스트를 포함하는 사각형 그리기
    draw.rectangle(text_bbox, fill="black")

    # 텍스트 그리기
    draw.text((0, 0), text, font=font, fill="white")

    # PyQt에서 사용할 수 있도록 QPixmap 객체로 변환
    img = img.convert("RGB")

    return img


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = EfficientNetVisualizer()
    ex.show()
    sys.exit(app.exec_())
