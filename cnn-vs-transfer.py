from pathlib import Path
import copy
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parent
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))     # N, 32, 30, 30
        x = self.pool(x)                 # N, 32, 15, 15
        x = self.relu(self.conv2(x))     # N, 64, 13, 13
        x = self.pool(x)                 # N, 64, 6, 6
        x = self.relu(self.conv3(x))     # N, 64, 4, 4
        x = torch.flatten(x, 1) # N, 1024
        x = self.relu(self.fc1(x))       # N, 64
        return self.fc2(self.dropout(x)) # N, 10 logits


class TransferResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = resnet18(weights=WEIGHTS)
        self.network.fc = nn.Linear(512, 10)
        self.network.requires_grad_(False)
        self.network.fc.requires_grad_(True)

    def enable_finetuning(self):
        self.network.layer4.requires_grad_(True)
        for layer in self.network.layer4.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        for layer in self.network.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self

    def forward(self, x):
        return self.network(x)


def set_seed():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(42)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def stratified_split(targets):
    rng = np.random.default_rng(6202)
    targets = np.asarray(targets)
    train, validation = [], []
    for class_id in range(10):
        indices = rng.permutation(np.flatnonzero(targets == class_id))
        validation.extend(indices[:500])
        train.extend(indices[500:])
    return rng.permutation(train), rng.permutation(validation)


def make_transform(pretrained, augment):
    operations = []
    if augment:
        operations.append(transforms.RandomHorizontalFlip(p=0.5))
    if pretrained:
        operations.append(WEIGHTS.transforms())
    else:
        operations.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transforms.Compose(operations)


def make_loader(dataset, shuffle):
    return DataLoader(dataset, batch_size=64, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(42))


def make_loaders(raw_train, raw_test, train_indices, validation_indices, pretrained, augment):
    train = copy.copy(raw_train)
    validation = copy.copy(raw_train)
    test = copy.copy(raw_test)
    train.transform = make_transform(pretrained, augment)
    validation.transform = test.transform = make_transform(pretrained, False)
    return (
        make_loader(Subset(train, train_indices), True),
        make_loader(Subset(validation, validation_indices), False),
        make_loader(test, False),
    )


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    samples = 0
    synchronize(device)
    started = time.perf_counter()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach() * labels.size(0)
        correct += (logits.detach().argmax(1) == labels).sum()
        samples += labels.size(0)
    synchronize(device)
    seconds = time.perf_counter() - started
    return {"loss": loss_sum.item() / samples,
            "accuracy": 100 * correct.item() / samples, "seconds": seconds}


def evaluate(model, loader, device):
    model.eval()
    loss_sum = torch.zeros((), device=device)
    confusion = torch.zeros(10, 10, dtype=torch.long, device=device)
    samples = 0
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += nn.functional.cross_entropy(logits, labels) * labels.size(0)
            prediction = logits.argmax(1)
            confusion += torch.bincount(labels * 10 + prediction, minlength=100).reshape(10, 10)
            samples += labels.size(0)
    synchronize(device)
    seconds = time.perf_counter() - started
    cm = confusion.cpu().numpy()
    true_positives = cm.diagonal()
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * true_positives, denominator, out=np.zeros(10), where=denominator > 0)
    recall = np.divide(true_positives, cm.sum(1), out=np.zeros(10), where=cm.sum(1) > 0)
    return {"loss": loss_sum.item() / samples,
            "accuracy": 100 * true_positives.sum().item() / samples,
            "macro_f1": float(f1.mean()), "recall": recall.tolist(),
            "confusion_matrix": cm.tolist(), "seconds": seconds}


def fit(model, train_loader, validation_loader, optimizer, device, epochs, name):
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    train_seconds = 0.0
    for epoch in range(1, epochs + 1):
        train = train_epoch(model, train_loader, optimizer, device)
        validation = evaluate(model, validation_loader, device)
        if not np.isfinite(train["loss"]) or not np.isfinite(validation["loss"]):
            raise RuntimeError("Loss is not finite; training stopped.")
        train_seconds += train["seconds"]
        for key in ("loss", "accuracy"):
            history[key].append(train[key])
            history["val_" + key].append(validation[key])
        print(f"{name} [{epoch}/{epochs}]: loss={train['loss']:.4f}, "
              f"val_loss={validation['loss']:.4f}, val_acc={validation['accuracy']:.2f}%", flush=True)
    return {"epochs": epochs, "history": history, "train_seconds": train_seconds,
            "validation_accuracy": validation["accuracy"]}


def record_result(model, metrics, test_loader, device, approach, augmentation, results, head_seconds=0):
    test = evaluate(model, test_loader, device)
    result = dict(metrics, approach=approach, augmentation=augmentation, test=test,
                  parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  total_train_seconds=metrics["train_seconds"] + head_seconds)
    results.append(result)
    print(f"Test: {test['accuracy']:.2f}%, F1={test['macro_f1']:.4f}; "
          f"training stage={metrics['train_seconds']:.1f}s, "
          f"training including head={result['total_train_seconds']:.1f}s; "
          f"test evaluation={test['seconds']:.1f}s\n", flush=True)


def save_results(results, device):
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    report = {
        "experiment": {
            "dataset": "CIFAR-10", "train_size": 45000, "validation_size": 5000, "test_size": 10000,
            "seed": 42, "split_seed": 6202, "batch_size": 64, "device": str(device),
            "weights": "ResNet18_Weights.IMAGENET1K_V1",
            "cnn_input": "32x32; normalize mean=std=0.5 per channel",
            "resnet_input": "resize 256, center crop 224, ImageNet normalization",
            "optimizer": "Adam; CNN/head LR=0.001; fine-tuning layer4 LR=0.00001, head LR=0.0001",
            "protocol": "One training per approach/augmentation, final epoch, no early stopping. "
                        "Fine-tuning continues the trained frozen model; BatchNorm stays frozen.",
            "training_time": "Sum of training epochs, including loading/transforms/optimization; "
                             "excludes validation, test, downloads and plotting. Fine-tuning total includes the head stage.",
            "test_time": "One evaluation of 10000 images in batches of 64; includes loading, "
                         "preprocessing, model inference and metric calculations. Not single-image model latency.",
        },
        "results": results,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), layout="constrained")
    for index, result in enumerate(results):
        row, column = index % 3, (index // 3) * 2
        history = result["history"]
        epochs = range(1, result["epochs"] + 1)
        for offset, metric in enumerate(("loss", "accuracy")):
            ax = axes[row, column + offset]
            ax.plot(epochs, history[metric], label="Train")
            ax.plot(epochs, history["val_" + metric], label="Validation")
            ax.set(title=f"{result['approach']} / {result['augmentation']}", xlabel="Epoch",
                   ylabel="Cross-entropy" if metric == "loss" else "Accuracy (%)")
            ax.set_ylim(bottom=0)
            if metric == "accuracy":
                ax.set_ylim(0, 100)
            ax.grid(alpha=0.2)
            ax.legend()
    fig.savefig(output / "training.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), layout="constrained")
    for ax, result in zip(axes.flat, results):
        cm = np.asarray(result["test"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel="Predicted class", ylabel="Actual class",
               title=f"{result['approach']} / {result['augmentation']}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        for i in range(10):
            for j in range(10):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.savefig(output / "confusion.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    labels = [f"{r['approach']}\n{r['augmentation']}" for r in results]
    values = (
        ([r["test"]["accuracy"] for r in results], "Test accuracy (%)"),
        ([r["trainable_parameters"] for r in results], "Trainable parameters (log scale)"),
        ([r["total_train_seconds"] for r in results], "Training time (seconds)\nIncluding prerequisite head training"),
        ([r["test"]["seconds"] for r in results], "Test evaluation time (seconds)\nIncluding preprocessing and metrics"),
    )
    for ax, (numbers, title) in zip(axes.flat, values):
        ax.bar(range(6), numbers)
        ax.set_xticks(range(6), labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylim(0, 100)
    axes[0, 1].set_yscale("log")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)

    winner = max(results, key=lambda r: r["validation_accuracy"])
    worst_class = int(np.argmin(winner["test"]["recall"]))
    print(f"Highest validation accuracy: {winner['approach']} / {winner['augmentation']}. "
          f"Its test accuracy: {winner['test']['accuracy']:.2f}%.")
    print(f"Lowest test recall for this model: {CLASSES[worst_class]} "
          f"({100 * winner['test']['recall'][worst_class]:.2f}%).")
    for baseline, augmented in zip(results[:3], results[3:]):
        difference = augmented["validation_accuracy"] - baseline["validation_accuracy"]
        print(f"Horizontal flip, {baseline['approach']}: validation accuracy change {difference:+.2f} pp.")
    print("Use the small CNN when model size and inference cost matter; "
          "frozen features when training resources are limited; "
          "fine-tuning when validation gains justify the additional training.")
    print("One seed was used; results do not measure variation between training runs. "
          "Input sizes and training budgets differ. Test time includes preprocessing.")
    print(f"Saved results.json, training.png, confusion.png and comparison.png to {output}")


def main():
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}. Fixed experiment: CNN 20 epochs; frozen head 5; fine-tuning 5 more.", flush=True)
    torch.hub.set_dir(str(ROOT / "data" / "torch-hub"))
    raw_train = datasets.CIFAR10(ROOT / "data", train=True, download=True)
    raw_test = datasets.CIFAR10(ROOT / "data", train=False, download=True)
    train_indices, validation_indices = stratified_split(raw_train.targets)
    results = []

    for augment in (False, True):
        augmentation = "horizontal flip" if augment else "no augmentation"
        train, validation, test = make_loaders(
            raw_train, raw_test, train_indices, validation_indices, False, augment)
        set_seed()
        model = CNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        metrics = fit(model, train, validation, optimizer, device, 20, f"CNN / {augmentation}")
        record_result(model, metrics, test, device, "CNN", augmentation, results)
        del model, optimizer

        train, validation, test = make_loaders(
            raw_train, raw_test, train_indices, validation_indices, True, augment)
        set_seed()
        model = TransferResNet().to(device)
        optimizer = torch.optim.Adam(model.network.fc.parameters(), lr=0.001)
        head_metrics = fit(model, train, validation, optimizer, device, 5, f"Frozen ResNet / {augmentation}")
        record_result(model, head_metrics, test, device, "Frozen ResNet", augmentation, results)

        model.enable_finetuning()
        optimizer = torch.optim.Adam([
            {"params": [p for p in model.network.layer4.parameters() if p.requires_grad], "lr": 0.00001},
            {"params": model.network.fc.parameters(), "lr": 0.0001},
        ])
        set_seed()
        train = make_loader(train.dataset, True)
        metrics = fit(model, train, validation, optimizer, device, 5, f"Fine-tuned ResNet / {augmentation}")
        record_result(model, metrics, test, device, "Fine-tuned ResNet", augmentation,
                      results, head_metrics["train_seconds"])
        del model, optimizer
        if device.type == "mps":
            torch.mps.empty_cache()

    save_results(results, device)


if __name__ == "__main__":
    main()
