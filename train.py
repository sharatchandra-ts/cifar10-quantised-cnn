import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from quant import set_fake_quant
from utils.config import load_config


def train(trainloader, valloader, net, device: torch.device):
    config = load_config()
    train_cfg = config.train  # type: ignore

    print(f"Training golden model on device: {device}")
    net.to(device)

    criterion = nn.CrossEntropyLoss()

    # SGD + momentum + weight decay — standard recipe for CIFAR-10
    # lr read from config so it's not hardcoded here
    optimizer = optim.SGD(
        net.parameters(), lr=train_cfg.learning_rate, momentum=0.9, weight_decay=5e-4
    )

    # CosineAnnealingLR: smoothly decays lr from learning_rate → ~0 over all epochs
    # T_max = total epochs, step ONCE PER EPOCH (not per batch)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)

    best_val_acc = 0.0
    best_ckpt_path = "./models/best_checkpoint.pt"

    for epoch in range(train_cfg.epochs):
        # ── Training pass ────────────────────────────────────────────────
        net.train()
        set_fake_quant(net, enabled=True)
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = criterion(net(inputs), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                # scheduler.get_last_lr() is safe here — scheduler hasn't stepped
                # yet this epoch but was stepped at end of previous epoch.
                # Use optimizer param_groups for epoch 1 safety.
                current_lr = optimizer.param_groups[0]["lr"]
                # print(
                #     f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}  "
                #     f"lr: {current_lr:.5f}"
                # )
                running_loss = 0.0

        # Step scheduler once per epoch AFTER the training loop
        scheduler.step()

        # ── Validation pass ──────────────────────────────────────────────
        if valloader:
            net.eval()
            set_fake_quant(net, enabled=True)  # measure quantized val accuracy

            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in valloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    val_loss += criterion(outputs, labels).item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += predicted.eq(labels).sum().item()

            epoch_val_loss = val_loss / total_val
            epoch_val_acc = 100.0 * correct_val / total_val
            current_lr = scheduler.get_last_lr()[0]

            print(
                f"[Epoch {epoch + 1}/{train_cfg.epochs}] "
                f"Val Loss: {epoch_val_loss:.4f} | "
                f"Val Acc: {epoch_val_acc:.2f}%  "
                f"lr: {current_lr:.5f}"
            )

            # Save best checkpoint during training
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                import os

                os.makedirs("models", exist_ok=True)
                torch.save(net.state_dict(), best_ckpt_path)
                print(f"  ↑ New best: {best_val_acc:.2f}% — saved to {best_ckpt_path}")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}%")
