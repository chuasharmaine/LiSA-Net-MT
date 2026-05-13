import re
import matplotlib.pyplot as plt
from collections import defaultdict

log_files = {
    "BreastCancerMT": "./pretrain/multitask/BreastCancerMT_ISIC-2018/log.txt",
    "MBDCNN": "./pretrain/multitask/MBDCNN_ISIC-2018/log.txt",
    "LiSANetMT": "./pretrain/multitask/LiSANetMT_ISIC-2018_NormMT/log.txt",
}

def get(line, key):
    match = re.search(rf"{key}:([0-9]*\.?[0-9]+)", line)
    return float(match.group(1)) if match else None

def smooth(values, alpha=0.3):
    """Exponential Moving Average smoothing"""
    if not values:
        return values

    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed

data = defaultdict(lambda: defaultdict(list))

for model, path in log_files.items():
    with open(path, "r") as f:
        for line in f:

            epoch_match = re.search(r"epoch:\[(\d+)/", line)
            if not epoch_match:
                continue

            epoch = int(epoch_match.group(1))

            metrics = {
                "DSC": get(line, "valid_DSC"),
                "IoU": get(line, "valid_IoU"),
                "ACC": get(line, "valid_ACC_cls"),
                "AUC": get(line, "valid_AUC"),
                "F1": get(line, "valid_F1"),
            }

            for k, v in metrics.items():
                if v is not None:
                    data[model][k].append((epoch, v))

for model in data:
    for metric in data[model]:
        data[model][metric].sort(key=lambda x: x[0])

def plot_metric(metric, ylabel, alpha=0.3):
    plt.figure()

    for model in data:
        if metric not in data[model]:
            continue

        epochs = [e for e, v in data[model][metric]]
        values = [v for e, v in data[model][metric]]

        values_smooth = smooth(values, alpha)

        plt.plot(epochs, values_smooth, label=model)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Comparison Across Models")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_metric("DSC", "Dice Coefficient (DSC)")
plot_metric("IoU", "Intersection over Union (IoU)")
plot_metric("ACC", "Classification Accuracy")
plot_metric("AUC", "AUC-ROC")
plot_metric("F1", "F1 Score")


