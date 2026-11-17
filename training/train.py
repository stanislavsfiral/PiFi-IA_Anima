"""
Advanced training script for Sfiral Ternary Copilot.
Parses real JSONL session logs from ai_memory/ to train on user's actual 3D models.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Автоматическое добавление корня проекта в путь Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quant.ternarize import TernaryLinear


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 16
    lr: float = 5e-4
    t: float = 0.7
    save_dir: str = "./models"
    model_name: str = "sfiral_ternary_copilot.pt"


class SfiralTernaryNet(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=3, t=0.7):
        super().__init__()
        self.fc1 = TernaryLinear(input_dim, hidden_dim, t=t)
        self.relu = nn.ReLU()
        self.fc2 = TernaryLinear(hidden_dim, output_dim, t=t)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def load_real_dataset_from_logs():
    """Считывает реальные данные узлов из логов конструктора в ai_memory/"""
    log_file = os.path.join("ai_memory", "ai_timeline_log.jsonl")
    features_list = []
    labels_list = []

    if os.path.exists(log_file):
        print(f"Загрузка сессий из {log_file}...")
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    session = json.loads(line)
                    nodes = session.get("nodes", [])
                    for node in nodes:
                        p = node.get("params", {})
                        angles = p.get("angles", [0, 0, 0])
                        stretch = p.get("stretch", 0.7778)

                        # Вектор признаков узла: X, Y, Z, RotX, RotY, RotZ, stretch, active_gate_marker
                        x = float(node.get("x", 0.0))
                        y = float(node.get("y", 0.0))
                        z = float(node.get("z", 0.0))

                        features_list.append(
                            [
                                x,
                                y,
                                z,
                                float(angles[0]),
                                float(angles[1]),
                                float(angles[2]),
                                float(stretch),
                                1.0,
                            ]
                        )

                        # Метка класса (тип квантового состояния / вентиля на основе углов)
                        gate_type = p.get("activeGate", "ROUTER_SWAP")
                        label = (
                            0
                            if gate_type == "ROUTER_SWAP"
                            else (1 if gate_type == "SCALE_CORRECTOR" else 2)
                        )
                        labels_list.append(label)
        except Exception as e:
            print(f"Ошибка чтения логов: {e}")

    # Если логов пока мало или файл пустой, добавляем синтетические примеры
    if len(features_list) < 10:
        print(
            "Логи пока содержат мало данных. Дополняем базовым синтетическим датасетом Сфирали..."
        )
        import numpy as np

        synth_x = np.random.randn(200, 8).tolist()
        synth_y = np.random.randint(0, 3, size=200).tolist()
        features_list.extend(synth_x)
        labels_list.extend(synth_y)

    return torch.tensor(features_list, dtype=torch.float32), torch.tensor(
        labels_list, dtype=torch.long
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train Sfiral Ternary Model on Real Logs"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    os.makedirs(cfg.save_dir, exist_ok=True)

    X_train, Y_train = load_real_dataset_from_logs()
    print(f"Датасет собран: всего примеров (узлов) — {len(X_train)}")

    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = SfiralTernaryNet(t=cfg.t)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Запуск обучения тернарной модели на {cfg.epochs} эпох...")
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Эпоха {epoch+1}/{cfg.epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}"
            )

    ckpt_path = os.path.join(cfg.save_dir, cfg.model_name)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
    print(f"Модель успешно переобучена на ваших данных и сохранена -> {ckpt_path}")


if __name__ == "__main__":
    main()