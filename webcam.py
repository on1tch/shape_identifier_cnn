import cv2
import torch
import numpy as np

from PIL import Image
from train.model import ShapeCNN_top
from train.dataset import eval_transform



CLASSES = ['circle', 'diamond', 'hexagon', 'oval', 'pentagon', 'rectangle', 'square', 'star', 'trapezoid', 'triangle']

MODEL_PATH = "models/shape_CNN_10type_irregular.pth"

CONFIDENCE_THRESHOLD = 0.55
MIN_CONTOUR_AREA = 500


def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ShapeCNN_top().to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Модель загружена: {device}")

    return model, device


def find_shapes(frame):
    """Находит контуры фигур на светлом фоне."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Предполагаем, что фигуры темнее фона
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel
    )

    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Отсекаем слишком маленькие области
        if w < 20 or h < 20:
            continue

        boxes.append((x, y, w, h))

    return boxes


def classify_crop(model, device, frame, box):
    """Классифицирует одну найденную фигуру."""

    x, y, w, h = box
    image_h, image_w = frame.shape[:2]

    # Делаем квадратную область с небольшим отступом
    side = int(max(w, h) * 1.2)
    side = max(side, 32)

    center_x = x + w // 2
    center_y = y + h // 2

    x1 = max(0, center_x - side // 2)
    y1 = max(0, center_y - side // 2)
    x2 = min(image_w, x1 + side)
    y2 = min(image_h, y1 + side)

    # Если у границы кадра квадрат получился обрезан,
    # дополняем его фоном
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None, 0.0

    top = max(0, side // 2 - center_y)
    left = max(0, side // 2 - center_x)

    bottom = max(0, side - crop.shape[0] - top)
    right = max(0, side - crop.shape[1] - left)

    crop = cv2.copyMakeBorder(
        crop,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    # OpenCV использует BGR, а PIL — RGB
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)

    tensor = eval_transform(pil_image).unsqueeze(0)
    tensor = tensor.to(device)

    with torch.inference_mode():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

        confidence, predicted_idx = probabilities.max(dim=0)

    confidence = confidence.item()
    predicted_idx = predicted_idx.item()

    return CLASSES[predicted_idx], confidence


def main():
    model, device = load_model()

    camera = cv2.VideoCapture(1)

    if not camera.isOpened():
        print("Не удалось открыть веб-камеру")
        return

    print("Нажми Q, чтобы выйти.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Не удалось получить кадр")
                break

            boxes = find_shapes(frame)

            for box in boxes:
                x, y, w, h = box

                result = classify_crop(
                    model, device, frame, box
                )

                if result is None:
                    continue

                label, confidence = result

                if confidence < CONFIDENCE_THRESHOLD:
                    text = f"Unknown {confidence:.0%}"
                    color = (0, 165, 255)
                else:
                    text = f"{label} {confidence:.0%}"
                    color = (0, 255, 0)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2,
                )

                text_y = max(y - 10, 25)

                cv2.putText(
                    frame,
                    text,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Shape Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()