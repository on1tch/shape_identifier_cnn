import os
import random
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, transforms
import math
from PIL import ImageDraw
from torch.utils.data import Subset


import random
import math


def irregular_polygon(cx, cy, radius_x, radius_y, sides,
                      angle_offset=0, jitter=0.12):
    """Создаёт слегка неправильный многоугольник."""
    points = []

    for i in range(sides):
        angle = 2 * math.pi * i / sides + angle_offset

        # У каждой вершины своё небольшое смещение.
        rx = radius_x * random.uniform(1 - jitter, 1 + jitter)
        ry = radius_y * random.uniform(1 - jitter, 1 + jitter)

        x = cx + math.cos(angle) * rx
        y = cy + math.sin(angle) * ry

        points.append((int(x), int(y)))

    return points


def irregular_rectangle(x1, y1, x2, y2, jitter=0.10):
    """Четырёхугольник с немного неодинаковыми сторонами."""
    width = x2 - x1
    height = y2 - y1

    points = [
        (x1 + random.uniform(-jitter, jitter) * width,
         y1 + random.uniform(-jitter, jitter) * height),

        (x2 + random.uniform(-jitter, jitter) * width,
         y1 + random.uniform(-jitter, jitter) * height),

        (x2 + random.uniform(-jitter, jitter) * width,
         y2 + random.uniform(-jitter, jitter) * height),

        (x1 + random.uniform(-jitter, jitter) * width,
         y2 + random.uniform(-jitter, jitter) * height),
    ]

    return [(int(x), int(y)) for x, y in points]


train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomAffine(
        degrees=25,
        translate=(0.10, 0.10),
        scale=(0.80, 1.20),
        fill=(255, 255, 255),
    ),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def draw_polygon(draw, points, color, filled, outline_width):
    if filled:
        draw.polygon(points, fill=color)
    else:
        draw.polygon(
            points,
            outline=color,
            width=outline_width,
        )


import math



def draw_shape(
    draw,
    shape_type,
    x1,
    y1,
    size,
    color,
    width,
    height,
    filled=True,
    outline_width=3,
):
    x2 = x1 + size
    y2 = y1 + size

    cx = x1 + size / 2
    cy = y1 + size / 2

    # Заливка может быть чёрной, но контур всегда остаётся чётким.
    fill_color = color if filled else (255, 255, 255)
    outline_color = color

    # Слегка неровный круг или овал.
    if shape_type in ("circle", "oval"):
        if shape_type == "circle":
            rx = size * random.uniform(0.44, 0.50)
            ry = size * random.uniform(0.44, 0.50)
        else:
            rx = size * random.uniform(0.44, 0.50)
            ry = size * random.uniform(0.29, 0.39)

        points = irregular_polygon(
            cx, cy, rx, ry,
            sides=48,
            angle_offset=random.uniform(-0.1, 0.1),
            jitter=0.035,
        )

        draw_polygon(
            draw, points, outline_color,
            filled, outline_width
        )

    # Прямоугольник с немного неровными углами.
    elif shape_type == "rectangle":
        w = min(width, size * 1.15)
        h = min(height, size * 1.15)

        points = irregular_rectangle(
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
            jitter=0.07,
        )

        draw_polygon(
            draw, points, outline_color,
            filled, outline_width
        )

    # Квадрат: стороны примерно одинаковые.
    elif shape_type == "square":
        points = irregular_rectangle(
            x1, y1, x2, y2,
            jitter=0.045,
        )

        draw_polygon(
            draw, points, outline_color,
            filled, outline_width
        )

    # Треугольник.
    elif shape_type == "triangle":
        points = irregular_polygon(
            cx, cy,
            size * 0.48,
            size * 0.46,
            sides=3,
            angle_offset=-math.pi / 2,
            jitter=0.10,
        )

        draw_polygon(
            draw, points, outline_color,
            filled, outline_width
        )

    # Ромб.
    elif shape_type == "diamond":
        points = irregular_polygon(
            cx, cy,
            size * random.uniform(0.40, 0.49),
            size * random.uniform(0.40, 0.49),
            sides=4,
            angle_offset=-math.pi / 2,
            jitter=0.10,
        )

        draw_polygon(
            draw, points, outline_color,
            filled, outline_width
        )

    # Пятиугольник.
    elif shape_type == "pentagon":
        points = irregular_polygon(
            cx, cy,
            size * 0.46,
            size * 0.46,
            sides=5,
            angle_offset=-math.pi / 2,
            jitter=0.10,
        )

        draw_polygon(
            draw, points, outline_color,
            filled, outline_width
        )

    # Шестиугольник.
    elif shape_type == "hexagon":
        points = irregular_polygon(
            cx, cy,
            size * 0.46,
            size * 0.46,
            sides=6,
            angle_offset=random.uniform(-0.2, 0.2),
            jitter=0.09,
        )

        draw_polygon(
            draw, points, outline_color,
            filled, outline_width
        )

    # Звезда с разными лучами.
    elif shape_type == "star":
        points = []

        for j in range(10):
            angle = math.pi * j / 5 - math.pi / 2
            angle += random.uniform(-0.045, 0.045)

            if j % 2 == 0:
                radius = size * random.uniform(0.40, 0.50)
            else:
                radius = size * random.uniform(0.19, 0.29)

            points.append((
                int(cx + radius * math.cos(angle)),
                int(cy + radius * math.sin(angle)),
            ))

        draw_polygon(
            draw, points, outline_color,
            filled, outline_width
        )

    # Трапеция с неодинаковыми основаниями и наклонными сторонами.
    elif shape_type == "trapezoid":
        top_width = size * random.uniform(0.40, 0.75)
        bottom_width = size * random.uniform(0.75, 1.00)

        top_y = y1 + size * random.uniform(0.05, 0.25)
        bottom_y = y1 + size * random.uniform(0.80, 0.98)

        shift_top = random.uniform(-0.12, 0.12) * size
        shift_bottom = random.uniform(-0.08, 0.08) * size

        points = [
            (int(cx + shift_top - top_width / 2), int(top_y)),
            (int(cx + shift_top + top_width / 2), int(top_y)),
            (int(cx + shift_bottom + bottom_width / 2), int(bottom_y)),
            (int(cx + shift_bottom - bottom_width / 2), int(bottom_y)),
        ]

        draw_polygon(
            draw, points, outline_color,
            filled, outline_width
        )

    else:
        raise ValueError(f"Неизвестная фигура: {shape_type}")



def generate_dataset(output_dir="dataset", num_samples=1000, img_size=(128, 128)):
    import os
    import random
    from PIL import Image, ImageDraw

    shapes = [
        "circle", "diamond", "hexagon", "oval", "pentagon",
        "rectangle", "square", "star", "trapezoid", "triangle"
    ]

    # Для чистого эксперимента лучше использовать новую папку.
    for shape in shapes:
        os.makedirs(os.path.join(output_dir, shape), exist_ok=True)

    shape_types = shapes * (num_samples // len(shapes))
    shape_types += random.choices(
        shapes,
        k=num_samples % len(shapes)
    )
    random.shuffle(shape_types)

    for i, shape_type in enumerate(shape_types):
        image = Image.new("RGB", img_size, "white")
        draw = ImageDraw.Draw(image)

        # Достаточно крупные фигуры с отступом от краёв.
        margin = 20
        max_size = min(img_size) - 2 * margin
        size = random.randint(int(max_size * 0.55), max_size)

        x1 = random.randint(margin, img_size[0] - margin - size)
        y1 = random.randint(margin, img_size[1] - margin - size)

        # Чёрный контур, как в твоих рисунках Paint.
        color = (0, 0, 0)
        outline_width = random.randint(3, 8)

        # Сначала исключаем заливку, чтобы проверить простой случай.
        filled = random.random() < 0.20

        # Варьируем ширину и высоту, но без экстремальных пропорций.
        width = random.randint(int(size * 0.8), size)
        height = random.randint(int(size * 0.8), size)

        draw_shape(
            draw=draw,
            shape_type=shape_type,
            x1=x1,
            y1=y1,
            size=size,
            color=color,
            width=width,
            height=height,
            filled=filled,
            outline_width=outline_width,
        )

        filename = f"{i}.png"
        image.save(os.path.join(output_dir, shape_type, filename))


# def create_dataset(output_dir="dataset"):
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#     ])
#
#     dataset = datasets.ImageFolder(
#         root=output_dir,
#         transform=transform,
#     )
#     test_size = int(len(dataset) * 0.15)
#     val_size = int(len(dataset) * 0.15)
#     train_size = len(dataset) - val_size - test_size
#
#     train_dataset, val_dataset, test_dataset = random_split(
#         dataset,
#         [train_size, val_size, test_size],
#         generator=torch.Generator().manual_seed(42),
#     )
#
#     return train_dataset, val_dataset, test_dataset

def create_dataset(output_dir="dataset"):
    base_dataset = datasets.ImageFolder(
        root=output_dir,
        transform=None,
    )

    n = len(base_dataset)

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n, generator=generator).tolist()

    test_size = int(n * 0.15)
    val_size = int(n * 0.15)

    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    train_base = datasets.ImageFolder(
        root=output_dir,
        transform=train_transform,
    )

    eval_base = datasets.ImageFolder(
        root=output_dir,
        transform=eval_transform,
    )

    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(eval_base, val_indices)
    test_dataset = Subset(eval_base, test_indices)

    return train_dataset, val_dataset, test_dataset

def create_dataloader(train_dataset, val_dataset, test_dataset):
    generator = torch.Generator().manual_seed(42)
    common_params = {
        "batch_size": 256,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
    }
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_params,
    )

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_params,
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_params,
    )


    return train_dataloader, val_dataloader, test_dataloader
