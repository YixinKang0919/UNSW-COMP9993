import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# ================== CONFIG ==================
CSV_PATH = "new_multi_radar_features.csv"   # 你的数据文件
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 100
PATIENCE = 10

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15   # test = 1 - train - val

LAMBDA_REG = 1.0
LAMBDA_CLS = 0.3   # 分类损失权重（可调小一点）

LOG_DIR = "MultiRadarSelector/runs"
MODEL_DIR = "MultiRadarSelector"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
OPT_PATH   = os.path.join(MODEL_DIR, "best_optimizer.pt")
SCALER_X_PATH = os.path.join(MODEL_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(MODEL_DIR, "scaler_Y.pkl")

torch.manual_seed(42)
np.random.seed(42)
# ============================================


# =============== 1. Load raw CSV =================
df = pd.read_csv(CSV_PATH, index_col=False, header=0)

required_cols = {"exp_idx", "idx", "direction", "hr", "rr", "radar_idx"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"缺少必要列: {missing}")

# =============== 2. 聚合为样本 (exp_idx, idx) =================
"""
每一行 = 某次实验某一段(idx)下：
  - 一个 radar_idx
  - 一个 chirp_idx
  - 一个 rx_tx_idx
  - 一段相位序列 phase_100..phase_399

我们要构造：
  每个样本 key = (exp_idx, idx)
  对于每个 key:
    - radar1_signal = 该 key 下所有 radar_idx==1 行的 phase 按行平均
    - radar2_signal = radar_idx==2 同理
    - hr, rr, direction 来自该 group（假设一致）
只保留同时存在 radar1 和 radar2 的样本。
"""

samples = []

# 只选纯 phase 列，排除 phase_diff
phase_cols = [c for c in df.columns if c.startswith("phase_") and not c.startswith("phase_diff")]
phase_cols = sorted(phase_cols)
L = len(phase_cols)
print(f"Using {L} phase cols:", phase_cols[:5], "...")

def dir_to_angle(d: int) -> float:
    """
    将你的方向编码映射成“该雷达相对被测人的夹角”，用于比较谁更正对胸口。
    这里按你之前说明来做一个合理假设（可以根据你真实定义微调）：

    0: 正对雷达       -> 0 度
    1: 左侧30 / 3:右30 -> 30 度
    2: 左侧60 / 4:右60 -> 60 度
    """
    mapping = {
        0: 0.0,
        1: 30.0,
        2: 60.0,
        3: 30.0,
        4: 60.0,
    }
    return mapping.get(int(d), 90.0)  # 未知给个大角度，表示很不正对

# ✅ 关键改动：只按 exp_idx 分组，不再用 idx
for exp_id, g in df.groupby("exp_idx"):
    # HR / RR：期望在同一个实验（两雷达）下是一致的
    hr_vals = g["hr"].unique()
    rr_vals = g["rr"].unique()
    if len(hr_vals) < 1 or len(rr_vals) < 1:
        print(f"Warning: missing hr/rr in exp_idx={exp_id}, skip.")
        continue

    # 若不唯一，取第一个并给出提醒（通常是标注冗余，不影响趋势）
    if len(hr_vals) != 1 or len(rr_vals) != 1:
        print(f"Warning: multiple hr/rr in exp_idx={exp_id}, using first one.")
    hr = float(hr_vals[0])
    rr = float(rr_vals[0])

    # 分别拿出两个雷达的全部行（所有 chirp_idx、rx_tx_idx）
    g_r1 = g[g["radar_idx"] == 1]
    g_r2 = g[g["radar_idx"] == 2]

    if len(g_r1) == 0 or len(g_r2) == 0:
        # 没有两个雷达就无法做“雷达选择”，这里先跳过
        print(f"Warning: exp_idx={exp_id} missing one radar, skip.")
        continue

    # 取每个雷达的 phase 矩阵并在 chirp/channel 维度上平均 -> 一条代表性波形
    r1_phase = g_r1[phase_cols].values.astype(np.float32)  # (N1, L)
    r2_phase = g_r2[phase_cols].values.astype(np.float32)  # (N2, L)
    if r1_phase.size == 0 or r2_phase.size == 0:
        continue

    r1_mean = r1_phase.mean(axis=0)  # (L,)
    r2_mean = r2_phase.mean(axis=0)  # (L,)

    # 每个雷达自己的 direction（应该在该 exp 内稳定）
    dir1_vals = g_r1["direction"].unique()
    dir2_vals = g_r2["direction"].unique()

    if len(dir1_vals) < 1 or len(dir2_vals) < 1:
        print(f"Warning: missing direction in exp_idx={exp_id}, skip.")
        continue

    if len(dir1_vals) != 1:
        print(f"Warning: multiple directions for radar1 in exp_idx={exp_id}, using first.")
    if len(dir2_vals) != 1:
        print(f"Warning: multiple directions for radar2 in exp_idx={exp_id}, using first.")

    d1 = int(dir1_vals[0])
    d2 = int(dir2_vals[0])

    ang1 = dir_to_angle(d1)
    ang2 = dir_to_angle(d2)

    # 哪个雷达“更正对”胸口 = 角度更小的那个
    if ang1 < ang2:
        best = 0  # radar1
    elif ang2 < ang1:
        best = 1  # radar2
    else:
        best = -1  # 两个一样好/信息不足，训练分类时忽略

    samples.append({
        "r1": r1_mean,
        "r2": r2_mean,
        "hr": hr,
        "rr": rr,
        "best_radar": best,
        "dir1": d1,
        "dir2": d2,
        "exp_idx": exp_id,
    })

print(f"Aggregated samples: {len(samples)}")
if len(samples) == 0:
    raise RuntimeError("聚合后没有可用样本，请检查 exp_idx/radar_idx 定义或数据内容。")



X_r1 = np.stack([s["r1"] for s in samples], axis=0)          # (N, L)
X_r2 = np.stack([s["r2"] for s in samples], axis=0)          # (N, L)
y_all = np.stack([[s["hr"], s["rr"]] for s in samples], 0)   # (N, 2)
best_radar_label = np.array([s["best_radar"] for s in samples], dtype=np.int64)

# 拼成 (N, 2L)，Dataset 里再拆成 (2,1,L)
X_all = np.concatenate([X_r1, X_r2], axis=1)

N = X_r1.shape[0]
print(f"Final usable samples: {N}, each with two-radar mean-phase.")


# =============== 3. derive best_radar_label from direction =================
"""
方向编码（你给的定义）：
0: 面向中间
1: 雷达在左侧30°
2: 左侧60°
3: 右侧30°
4: 右侧60°

这里做一个简单 heuristic：
- 1,2: 更靠近雷达1 -> best_radar = 0
- 3,4: 更靠近雷达2 -> best_radar = 1
- 0: 模糊，不参与分类 loss，用 -1 标记
"""


print("Best radar label counts:",
      {v: int((best_radar_label == v).sum()) for v in [-1, 0, 1]})


# =============== 4. Build X_all and split =================
# 拼成 (N, 2L)，后面 Dataset 再拆成 (2,1,L)
X_all = np.concatenate([X_r1, X_r2], axis=1)  # (N, 2L)
idx_all = np.arange(N)

# train / temp
train_idx, temp_idx = train_test_split(
    idx_all,
    train_size=TRAIN_RATIO,
    shuffle=True,
    random_state=42
)

# val / test
val_ratio_within_temp = VAL_RATIO / (1.0 - TRAIN_RATIO)
val_idx, test_idx = train_test_split(
    temp_idx,
    train_size=val_ratio_within_temp,
    shuffle=True,
    random_state=42
)

def split(arr):
    return arr[train_idx], arr[val_idx], arr[test_idx]

X_train_raw, X_val_raw, X_test_raw = split(X_all)
y_train_raw, y_val_raw, y_test_raw = split(y_all)
best_radar_train, best_radar_val, best_radar_test = split(best_radar_label)

print("train:", X_train_raw.shape,
      "val:", X_val_raw.shape,
      "test:", X_test_raw.shape)


# =============== 5. Normalization (fit on train only) =================
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train_raw)
X_val   = scaler_X.transform(X_val_raw)
X_test  = scaler_X.transform(X_test_raw)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train_raw)
y_val   = scaler_y.transform(y_val_raw)
y_test  = scaler_y.transform(y_test_raw)

joblib.dump(scaler_X, SCALER_X_PATH)
joblib.dump(scaler_y, SCALER_Y_PATH)
print(f"Saved scalers to {SCALER_X_PATH}, {SCALER_Y_PATH}")


# =============== 6. Dataset =================
class MultiRadarDataset(Dataset):
    """
    每个样本:
      输入 x: (2,1,L)  -> [radar1_mean_phase, radar2_mean_phase]
      输出 y: (2,)     -> 标准化后的 [HR, RR]
      radar_label: 0/1/-1 (用于分类分支的监督，-1 表示忽略)
    """
    def __init__(self, X, y, radar_label, seq_len):
        self.X = X
        self.y = y
        self.radar_label = radar_label
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_flat = self.X[idx]            # (2L,)
        y = self.y[idx]                 # (2,)
        rlabel = int(self.radar_label[idx])

        L = self.seq_len
        r1 = x_flat[:L]
        r2 = x_flat[L:]
        # (2,1,L)
        x = np.stack([r1, r2], axis=0)[:, None, :]

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y, rlabel


seq_len = L
train_dataset = MultiRadarDataset(X_train, y_train, best_radar_train, seq_len)
val_dataset   = MultiRadarDataset(X_val,   y_val,   best_radar_val,   seq_len)
test_dataset  = MultiRadarDataset(X_test,  y_test,  best_radar_test,  seq_len)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)


# =============== 7. Model: SharedLightCNN + radar selection + regression ===============
class SharedLightCNN(nn.Module):
    """
    轻量 1D-CNN，对单个雷达 (1, L) 提取 embedding h_r ∈ R^D
    """
    def __init__(self, in_channels=1, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(1),   # -> (B,64,1)
        )
        self.out_dim = out_dim

    def forward(self, x):
        # x: (B, 1, L)
        h = self.net(x).squeeze(-1)  # (B,64)
        return h


class MultiRadarSelectorReg(nn.Module):
    """
    输入: x ∈ R^{B, 2, 1, L}
    1) 共享CNN 编码 -> h1, h2
    2) 分类头对每个雷达输出 logit，softmax 得到权重
    3) 加权融合特征 -> 回归头输出 HR/RR
    """
    def __init__(self, num_radars=2, in_channels=1, d_model=64, output_dim=2):
        super().__init__()
        self.num_radars = num_radars
        self.encoder = SharedLightCNN(in_channels=in_channels, out_dim=d_model)

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.1),
            nn.Linear(d_model, 1)  # 每个雷达一个 score
        )

        self.reg_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        """
        x: (B, R=2, C=1, L)
        返回:
          y_pred: (B,2)
          cls_logits: (B,R)
          attn: (B,R)
        """
        B, R, C, L = x.shape
        assert R == self.num_radars

        # 展开 radar 维度，共享 CNN
        x_flat = x.view(B * R, C, L)        # (B*R,1,L)
        h_flat = self.encoder(x_flat)       # (B*R,d)
        h = h_flat.view(B, R, -1)           # (B,R,d)

        # 分类 logits
        cls_logits = self.cls_head(h).squeeze(-1)  # (B,R)

        # softmax 权重（哪个雷达更对）
        attn = torch.softmax(cls_logits, dim=1)    # (B,R)

        # 融合雷达特征
        h_fused = torch.sum(attn.unsqueeze(-1) * h, dim=1)  # (B,d)

        # 回归 HR/RR
        y_pred = self.reg_head(h_fused)            # (B,2)

        return y_pred, cls_logits, attn


model = MultiRadarSelectorReg(
    num_radars=2,
    in_channels=1,
    d_model=64,
    output_dim=2
).to(DEVICE)

criterion_reg = nn.SmoothL1Loss()        # Huber，比 MSE 稳定
criterion_cls = nn.CrossEntropyLoss(reduction="none")  # 手动 mask 掉 label=-1
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
writer = SummaryWriter(LOG_DIR)


# =============== 8. Training with early stopping =================
best_val_loss = float("inf")
patience_cnt = 0

for epoch in range(EPOCHS):
    # ---- train ----
    model.train()
    train_loss_sum = 0.0
    n_train = 0

    for xb, yb, rlabel in train_loader:
        xb = xb.to(DEVICE)           # (B,2,1,L)
        yb = yb.to(DEVICE)           # (B,2)
        rlabel = rlabel.to(DEVICE)   # (B,)

        optimizer.zero_grad()
        y_pred, cls_logits, attn = model(xb)

        # 回归损失
        loss_reg = criterion_reg(y_pred, yb)

        # 分类损失（只对 rlabel>=0 的样本）
        mask = (rlabel >= 0)
        if mask.any():
            cls_logits_masked = cls_logits[mask]        # (M,2)
            rlabel_masked = rlabel[mask]                # (M,)
            loss_cls_vec = criterion_cls(cls_logits_masked, rlabel_masked)
            loss_cls = loss_cls_vec.mean()
            loss = LAMBDA_REG * loss_reg + LAMBDA_CLS * loss_cls
        else:
            loss_cls = torch.tensor(0.0, device=DEVICE)
            loss = LAMBDA_REG * loss_reg

        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        train_loss_sum += loss.item() * bs
        n_train += bs

    train_loss = train_loss_sum / max(1, n_train)

    # ---- val ----
    model.eval()
    val_loss_sum = 0.0
    n_val = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for xb, yb, rlabel in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            rlabel = rlabel.to(DEVICE)

            y_pred, cls_logits, attn = model(xb)
            loss_reg = criterion_reg(y_pred, yb)

            mask = (rlabel >= 0)
            if mask.any():
                cls_logits_masked = cls_logits[mask]
                rlabel_masked = rlabel[mask]
                loss_cls_vec = criterion_cls(cls_logits_masked, rlabel_masked)
                loss_cls = loss_cls_vec.mean()
                loss_batch = LAMBDA_REG * loss_reg + LAMBDA_CLS * loss_cls
            else:
                loss_batch = LAMBDA_REG * loss_reg

            bs = xb.size(0)
            val_loss_sum += loss_batch.item() * bs
            n_val += bs

            all_true.append(yb.cpu().numpy())
            all_pred.append(y_pred.cpu().numpy())

    val_loss = val_loss_sum / max(1, n_val)
    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    mae_hr_std = mean_absolute_error(all_true[:, 0], all_pred[:, 0])
    mae_rr_std = mean_absolute_error(all_true[:, 1], all_pred[:, 1])

    print(f"Epoch {epoch+1}/{EPOCHS} "
          f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
          f"std-MAE(HR)={mae_hr_std:.4f} std-MAE(RR)={mae_rr_std:.4f}")

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("MAE_std/hr", mae_hr_std, epoch)
    writer.add_scalar("MAE_std/rr", mae_rr_std, epoch)

    # early stopping
    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        patience_cnt = 0
        torch.save(model.state_dict(), MODEL_PATH)
        torch.save(optimizer.state_dict(), OPT_PATH)
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print("Early stopping triggered.")
            break

writer.close()
print(f"Best val loss: {best_val_loss:.6f}")


# =============== 9. Testing (inverse transform, metrics) =================
best_model = MultiRadarSelectorReg(
    num_radars=2,
    in_channels=1,
    d_model=64,
    output_dim=2
).to(DEVICE)
best_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
best_model.eval()

all_true_std = []
all_pred_std = []

with torch.no_grad():
    for xb, yb, rlabel in test_loader:
        xb = xb.to(DEVICE)
        y_pred, cls_logits, attn = best_model(xb)
        all_true_std.append(yb.numpy())
        all_pred_std.append(y_pred.cpu().numpy())

all_true_std = np.concatenate(all_true_std, axis=0)
all_pred_std = np.concatenate(all_pred_std, axis=0)

# 反标准化
all_true = scaler_y.inverse_transform(all_true_std)
all_pred = scaler_y.inverse_transform(all_pred_std)

mae_hr = mean_absolute_error(all_true[:, 0], all_pred[:, 0])
mae_rr = mean_absolute_error(all_true[:, 1], all_pred[:, 1])
rmse_hr = np.sqrt(np.mean((all_true[:, 0] - all_pred[:, 0])**2))
rmse_rr = np.sqrt(np.mean((all_true[:, 1] - all_pred[:, 1])**2))
corr_hr = np.corrcoef(all_true[:, 0], all_pred[:, 0])[0, 1]
corr_rr = np.corrcoef(all_true[:, 1], all_pred[:, 1])[0, 1]

print("===== TEST (original space) =====")
print(f"MAE(HR):  {mae_hr:.3f}")
print(f"MAE(RR):  {mae_rr:.3f}")
print(f"RMSE(HR): {rmse_hr:.3f}, Corr(HR): {corr_hr:.3f}")
print(f"RMSE(RR): {rmse_rr:.3f}, Corr(RR): {corr_rr:.3f}")

# =============== 10. Plots =================
error = np.abs(all_true - all_pred)
errors_hr = np.sort(error[:, 0])
errors_rr = np.sort(error[:, 1])
cdf_hr = np.arange(1, len(errors_hr)+1) / len(errors_hr)
cdf_rr = np.arange(1, len(errors_rr)+1) / len(errors_rr)

plt.figure()
plt.plot(errors_hr, cdf_hr, label="HR")
plt.plot(errors_rr, cdf_rr, label="RR")
plt.xlabel("Absolute Error")
plt.ylabel("CDF")
plt.title("CDF of Absolute Errors")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "cdf_plot.png"), dpi=200)

plt.figure()
plt.scatter(all_true[:, 0], all_pred[:, 0], alpha=0.5)
min_hr = min(all_true[:, 0].min(), all_pred[:, 0].min())
max_hr = max(all_true[:, 0].max(), all_pred[:, 0].max())
plt.plot([min_hr, max_hr], [min_hr, max_hr], linestyle="--")
plt.xlabel("True HR")
plt.ylabel("Pred HR")
plt.title(f"HR Scatter (RMSE={rmse_hr:.2f}, r={corr_hr:.2f})")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "scatter_hr.png"), dpi=200)

plt.figure()
plt.scatter(all_true[:, 1], all_pred[:, 1], alpha=0.5)
min_rr = min(all_true[:, 1].min(), all_pred[:, 1].min())
max_rr = max(all_true[:, 1].max(), all_pred[:, 1].max())
plt.plot([min_rr, max_rr], [min_rr, max_rr], linestyle="--")
plt.xlabel("True RR")
plt.ylabel("Pred RR")
plt.title(f"RR Scatter (RMSE={rmse_rr:.2f}, r={corr_rr:.2f})")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "scatter_rr.png"), dpi=200)

print("Saved plots in", MODEL_DIR)
print("All done ✅")
