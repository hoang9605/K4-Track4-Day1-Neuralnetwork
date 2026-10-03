"""Vẽ lịch sử huấn luyện và so sánh các thí nghiệm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    cfg, history = result["cfg"], result["history"]
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for metric in ("train_loss", "val_loss"):
        axes[0].plot(epochs, history[metric], label=metric)
    for metric in ("val_acc", "val_macro_f1"):
        axes[1].plot(epochs, history[metric], label=metric)
    axes[2].plot(epochs, history["grad_norm"], label="grad_norm")
    for ax, label in zip(axes, ("Loss", "Score", "Gradient L2")):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
        if result["summary"]["best_epoch"]:
            ax.axvline(result["summary"]["best_epoch"], color="gray", linestyle="--", alpha=0.5)
    fig.suptitle(f"{cfg['exp_id']} | {cfg['optimizer']} lr={cfg['lr']} batch={cfg['batch']}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        ax.plot(result["history"]["epoch"], result["history"][metric], label=result["cfg"]["exp_id"])
    ax.set(xlabel="Epoch", ylabel=metric, title=title or metric)
    ax.legend()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
