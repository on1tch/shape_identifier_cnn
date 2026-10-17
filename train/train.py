import torch
import torch.nn as nn
import torch.optim as optim
from model import ShapeCNN, ShapeCNN_top
from dataset import (
    generate_dataset,
    create_dataset,
    create_dataloader,
)
import os
from sklearn.metrics import classification_report
import time


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp, scaler):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = y_batch.size(0)
        total_loss += loss.detach().item() * batch_size
        correct += (outputs.detach().argmax(dim=1) == y_batch).sum().item()
        total += batch_size

    return total_loss / total, correct / total




def evaluate(model, loader, criterion, device, use_amp=False):


    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(X)
                loss = criterion(outputs, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == y).sum().item()
            total += batch_size

    return total_loss / total, correct / total


# generate_dataset(output_dir="dataset_irregular", num_samples=100000)
def main():
    start_time = time.perf_counter()

    model = ShapeCNN_top()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0005
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=8,
        gamma=0.5
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ускорение операций с изображениями фиксированного размера
    torch.backends.cudnn.benchmark = True

    # Разрешаем TF32 на поддерживаемых GPU
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Устройство:", device)
    model = model.to(device)

    train_ds, val_ds, test_ds = create_dataset(output_dir="../dataset_irregular", )
    train_dl, val_dl, test_dl = create_dataloader(
        train_ds, val_ds, test_ds
    )

    print(train_ds.dataset.classes)
    epochs = 30

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_dl, criterion, optimizer,
            device, use_amp, scaler
        )

        val_loss, val_acc = evaluate(
            model, val_dl, criterion, device, use_amp
        )

        scheduler.step()
        elapsed_time = time.perf_counter() - start_time
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | "
            f"Time: {elapsed_time:.4f}"
        )

    test_loss, test_acc = evaluate(
        model, test_dl, criterion, device, use_amp
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2%}")

    # Создаем папку models, если она еще не существует
    os.makedirs("../models", exist_ok=True)

    # Путь для сохранения
    model_path = "../models/shape_CNN_10type_irregular.pth"

    # Сохраняем только веса (рекомендуемый способ в PyTorch)
    torch.save(model.state_dict(), model_path)
    print(f"Веса модели успешно сохранены в {model_path}")

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for X, y in test_dl:
            X = X.to(device, non_blocking=True)

            with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
            ):
                outputs = model(X)

            predictions = outputs.argmax(dim=1).cpu()

            all_predictions.extend(predictions.tolist())
            all_labels.extend(y.tolist())

    classes = train_ds.dataset.classes

    print(classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0,
    ))

    elapsed_time = time.perf_counter() - start_time
    print(f"Время выполнения: {elapsed_time:.4f} сек.")

if __name__ == "__main__":
    main()