import torch

from model import GoldenCNNModel
from quant import QuantBits, prepare_qat
from utils.config import load_config


def valrandom():
    # 1. Initialize your model and wrap it exactly how you do in benchmark.py
    config = load_config()
    net = GoldenCNNModel(config=config)
    prepare_qat(net, bits=QuantBits.INT4)
    net.load_state_dict(
        torch.load("./models/golden_model_qat_INT8.pt", weights_only=True)
    )
    net.eval()

    # 2. Generate completely random pixel values (garbage inputs)
    fake_images = torch.rand(100, 3 if not config.model.greyscale else 1, 32, 32)
    fake_labels = torch.randint(0, 10, (100,))  # Random target classes (0-9)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # 3. Create a fake dataloader tuple and evaluate it
    with torch.no_grad():
        outputs = net(fake_images)
        _, predicted = torch.max(outputs, 1)

        correct = (predicted.cpu() == fake_labels).sum().item()
        print(f"Accuracy on pure noise: {100 * correct / 100:.2f}%")


def val2():
    # Load your best checkpoint
    checkpoint = torch.load("./models/best_checkpoint.pt", map_location="cpu")
    # Pick a layer (e.g., Conv2 layer weights)
    weights = (
        checkpoint["state_dict"]["conv2.weight"]
        if "state_dict" in checkpoint
        else checkpoint["conv2.weight"]
    )

    # Flatten the tensor and find unique values
    unique_vals = torch.unique(weights)
    print(f"Number of unique floating-point values in layer: {len(unique_vals)}")
    print("Unique values:", unique_vals)


if __name__ == "__main__":
    val2()
