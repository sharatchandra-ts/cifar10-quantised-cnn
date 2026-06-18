import torch

from quant import set_fake_quant


def test(testloader, classes, net, device: torch.device, quantized: bool = False):
    """
    Evaluates the network performance on the test dataset.
    """
    # Make sure the network itself is in evaluation mode and on the correct device
    net.eval()
    net.to(device)

    set_fake_quant(net, enabled=quantized)

    # Quick Sample Prediction (First Batch)
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    images = images.to(device)

    with torch.no_grad():
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

    print(
        "GroundTruth: ",
        " ".join(f"{classes[labels[j]]:5s}" for j in range(min(4, len(labels)))),
    )
    print(
        "Predicted:   ",
        " ".join(f"{classes[predicted[j]]:5s}" for j in range(min(4, len(predicted)))),
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_accuracy = 100 * correct / total
    print(f"Accuracy of the network on the {total} test images: {final_accuracy:.2f} %")
    return final_accuracy
