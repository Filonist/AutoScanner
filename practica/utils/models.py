import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO


yolo_model = YOLO('yolov8n.pt')
maskrcnn_model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
maskrcnn_model.eval()

resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
])

def process_with_model(file_path: str, mode: str, file_id: str):
    """
    Обработка файла в зависимости от режима.
    Возвращает путь до результата (картинка с разметкой) и статистику.
    """
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError("Cannot read the image")

    if mode == 'detection':
        return detect_yolo(img, file_id)
    elif mode == 'segmentation':
        return segment_maskrcnn(img, file_id)
    elif mode == 'classification':
        return classify_resnet(img, file_id)
    else:
        raise ValueError("Unsupported mode")

def detect_yolo(img, file_id):
    results = yolo_model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    img_result = img.copy()
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_result, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img_result, f'{int(cls)}:{score:.2f}', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    result_path = f'results/{file_id}_det.jpg'
    cv2.imwrite(result_path, img_result)

    stats = {
        'objects_detected': len(boxes)
    }
    return result_path, stats

def segment_maskrcnn(img, file_id):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    transform = transforms.ToTensor()
    tensor_img = transform(pil_img)

    with torch.no_grad():
        predictions = maskrcnn_model([tensor_img])[0]

    masks = predictions['masks'].cpu().numpy()
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    threshold = 0.5

    img_result = img.copy()

    count = 0
    for i, score in enumerate(scores):
        if score > 0.7:
            mask = masks[i, 0]
            mask = (mask > threshold).astype(np.uint8) * 255
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            img_result[mask == 255] = img_result[mask == 255] * 0.5 + color * 0.5
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(img_result, (x1,y1), (x2,y2), color.tolist(), 2)
            count += 1

    result_path = f'results/{file_id}_seg.jpg'
    cv2.imwrite(result_path, img_result)

    stats = {
        'objects_segmented': count
    }
    return result_path, stats

def classify_resnet(img, file_id):
    h, w = img.shape[:2]
    side = min(h, w)
    center_crop = img[(h - side)//2:(h + side)//2, (w - side)//2:(w + side)//2]

    pil_img = Image.fromarray(cv2.cvtColor(center_crop, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(pil_img).unsqueeze(0)

    with torch.no_grad():
        outputs = resnet_model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    class_id = predicted.item()


    img_result = img.copy()
    cv2.putText(img_result, f'Class ID: {class_id}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    result_path = f'results/{file_id}_cls.jpg'
    cv2.imwrite(result_path, img_result)

    stats = {
        'predicted_class_id': class_id
    }
    return result_path, stats
