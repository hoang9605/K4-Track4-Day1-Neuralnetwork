"""Nạp dữ liệu CoverType, tách validation và chuẩn hoá đặc trưng.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    processed_dir = Path(processed_dir)
    with np.load(processed_dir / "train.npz") as train, np.load(processed_dir / "eval.npz") as eval_data:
        X_train_full, y_train_full = train["X"], train["y"]
        X_eval, y_eval, eval_row_id = eval_data["X"], eval_data["y"], eval_data["row_id"]
    for X, y in ((X_train_full, y_train_full), (X_eval, y_eval)):
        assert X.ndim == 2 and X.shape[1] == 54 and X.dtype == np.float32
        assert y.shape == (len(X),) and y.dtype == np.int64
        assert np.all((0 <= y) & (y < 7))
    assert eval_row_id.shape == (len(X_eval),) and eval_row_id.dtype == np.int64
    assert len(X_train_full) == 464_809 and len(X_eval) == 116_203
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    numeric = X_tr[:, :N_NUMERIC].astype(np.float64)
    mean = numeric.mean(axis=0)
    std = numeric.std(axis=0)
    return mean, np.where(std == 0, 1.0, std)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    result = X.copy()
    result[:, :N_NUMERIC] = (X[:, :N_NUMERIC] - mean) / std
    return result


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_full, y_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)
    arrays = {
        "X_tr": apply_standardizer(X_tr, mean, std),
        "y_tr": y_tr,
        "X_val": apply_standardizer(X_val, mean, std),
        "y_val": y_val,
        "X_eval": apply_standardizer(X_eval, mean, std),
        "y_eval": y_eval,
    }
    data = {name: torch.as_tensor(array, device=device) for name, array in arrays.items()}
    data["eval_row_id"] = eval_row_id
    majority_class = np.bincount(y_tr, minlength=7).argmax()
    majority_acc = np.mean(y_val == majority_class)
    print(f"train={len(y_tr)}, val={len(y_val)}, eval={len(y_eval)}")
    print(f"Đoán luôn lớp {majority_class}: val accuracy={majority_acc:.4f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    indices = (torch.randperm(len(X), device=X.device, generator=generator)
               if shuffle else torch.arange(len(X), device=X.device))
    for start in range(0, len(X), batch_size):
        index = indices[start:start + batch_size]
        yield X[index], y[index]
