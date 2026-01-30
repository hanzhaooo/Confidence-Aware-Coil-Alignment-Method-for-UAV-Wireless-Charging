# -*- coding: utf-8 -*-
"""
STM32Cube.AI friendly ONNX export with NEW confidence definition (MC consistency):

Input (float32, shape=(1,5)):
  [sx, sy, vavg, dsx, dsy]

Output (float32, shape=(1,4)):
  y = [dir_x, dir_y, alpha, c]

Confidence definition (training target):
  Run M stochastic predictions (dropout + small input noise) for same input:
    u_hat_j -> normalize -> u_tilde_j
  C = || mean_j(u_tilde_j) ||_2

Deployment:
  MCU does single forward pass. We train a confidence head c_head to regress C.

Export constraints for X-CUBE-AI 10.2.0:
- opset_version = 13
- ONNX ir_version forced to 9
- static shapes (no dynamic axes)
- single output tensor y (B,4)
"""

SIM_CSV_DEFAULT = r"R=140三维.csv"
EXP_CSV_DEFAULT = r"sss.csv"
OUT_ONNX_DEFAULT = r"alignnet_cubeai_mcconf.onnx"

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# 1) Parse Serial Debug CSV (逗号/空格兼容)
# -----------------------------
def parse_serial_debug(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if lines and ("PROC" in lines[0] or "proc" in lines[0]):
        lines = lines[1:]

    p_head_comma = re.compile(
        r"Loc\(x y\):,([-\d.]+),([-\d.]+),ADC\(V\):,([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)"
    )
    p_head_space = re.compile(
        r"Loc\(x y\):\s*([-\d.]+)\s+([-\d.]+)\s+ADC\(V\):\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    )
    p_tail_space = re.compile(r"^\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$")

    rows = []
    cur_x, cur_y = None, None

    for s in lines:
        m = p_head_comma.search(s)
        if m:
            cur_x, cur_y = float(m.group(1)), float(m.group(2))
            v1, v2, v3, v4 = map(float, m.group(3, 4, 5, 6))
            rows.append([cur_x, cur_y, v1, v2, v3, v4])
            continue

        m = p_head_space.search(s)
        if m:
            cur_x, cur_y = float(m.group(1)), float(m.group(2))
            v1, v2, v3, v4 = map(float, m.group(3, 4, 5, 6))
            rows.append([cur_x, cur_y, v1, v2, v3, v4])
            continue

        m2 = p_tail_space.match(s)
        if m2 and cur_x is not None:
            v1, v2, v3, v4 = map(float, m2.group(1, 2, 3, 4))
            rows.append([cur_x, cur_y, v1, v2, v3, v4])

    if not rows:
        raise RuntimeError("Parsed 0 rows. CSV format not matched.")

    df = pd.DataFrame(rows, columns=["x", "y", "v1", "v2", "v3", "v4"])
    df["vavg"] = df[["v1", "v2", "v3", "v4"]].mean(axis=1)
    return df


# -----------------------------
# 1b) 解析仿真 CSV（R=140三维.csv）
# -----------------------------
def parse_sim_csv(path: str) -> pd.DataFrame:
    """
    期望的列名（你当前文件的表头就是这种格式）：
      - x:  "DXX [mm]"
      - y:  "DYY [mm]"
      - v1: "L(RX,jc1) [uH]"
      - v2: "L(RX,jc2) [uH]"
      - v3: "L(RX,jc3) [uH]"
      - v4: "L(RX,jc4) [uH]"

    如果你的仿真列名有变化，只需要改下面 col_map 即可。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sim CSV not found: {path}")

    df = pd.read_csv(path)
    col_map = {
        "x": "DXX [mm]",
        "y": "DYY [mm]",
        "v1": "L(RX,jc1) [uH]",
        "v2": "L(RX,jc2) [uH]",
        "v3": "L(RX,jc3) [uH]",
        "v4": "L(RX,jc4) [uH]",
    }

    missing = [k for k, v in col_map.items() if v not in df.columns]
    if missing:
        raise RuntimeError(
            "仿真CSV缺少必要列: " + str(missing) + "\n" + "当前表头: " + str(list(df.columns))
        )

    out = pd.DataFrame(
        {
            "x": df[col_map["x"]].astype(float),
            "y": df[col_map["y"]].astype(float),
            "v1": df[col_map["v1"]].astype(float),
            "v2": df[col_map["v2"]].astype(float),
            "v3": df[col_map["v3"]].astype(float),
            "v4": df[col_map["v4"]].astype(float),
        }
    )
    out["vavg"] = out[["v1", "v2", "v3", "v4"]].mean(axis=1)
    return out


def build_grid(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["x", "y"], as_index=False)[["v1", "v2", "v3", "v4", "vavg"]].mean()


def finite_difference_gradient(grid: pd.DataFrame, field_col: str = "vavg") -> pd.DataFrame:
    F_map: Dict[Tuple[float, float], float] = {
        (row.x, row.y): float(getattr(row, field_col)) for row in grid.itertuples(index=False)
    }
    xs = np.sort(grid["x"].unique())
    ys = np.sort(grid["y"].unique())
    dx = np.median(np.diff(xs)) if len(xs) > 1 else 1.0
    dy = np.median(np.diff(ys)) if len(ys) > 1 else 1.0
    if not np.isfinite(dx) or dx == 0:
        dx = 1.0
    if not np.isfinite(dy) or dy == 0:
        dy = 1.0

    def getF(x, y):
        return F_map.get((x, y), None)

    grads = []
    for row in grid.itertuples(index=False):
        x, y = float(row.x), float(row.y)
        f0 = float(row.vavg)

        f_l = getF(x - dx, y)
        f_r = getF(x + dx, y)
        if f_l is not None and f_r is not None:
            dfdx = (f_r - f_l) / (2.0 * dx)
        elif f_r is not None:
            dfdx = (f_r - f0) / dx
        elif f_l is not None:
            dfdx = (f0 - f_l) / dx
        else:
            dfdx = 0.0

        f_d = getF(x, y - dy)
        f_u = getF(x, y + dy)
        if f_d is not None and f_u is not None:
            dfdy = (f_u - f_d) / (2.0 * dy)
        elif f_u is not None:
            dfdy = (f_u - f0) / dy
        elif f_d is not None:
            dfdy = (f0 - f_d) / dy
        else:
            dfdy = 0.0

        norm = float(np.sqrt(dfdx * dfdx + dfdy * dfdy) + 1e-12)
        gx, gy = dfdx / norm, dfdy / norm
        grads.append([x, y, gx, gy])

    gdf = pd.DataFrame(grads, columns=["x", "y", "gx", "gy"])
    return grid.merge(gdf, on=["x", "y"], how="left")


def compute_features(grid: pd.DataFrame) -> pd.DataFrame:
    """
    默认假设：v1=Left, v2=Right, v3=Down, v4=Up
    若线圈编号不同，改 sx/sy 两行。
    """
    out = grid.copy()
    out["sx"] = out["v2"] - out["v1"]
    out["sy"] = out["v4"] - out["v3"]
    out["vavg"] = out[["v1", "v2", "v3", "v4"]].mean(axis=1)

    out = out.sort_values(["y", "x"]).reset_index(drop=True)
    out["dsx"] = out["sx"].diff().fillna(0.0)
    out["dsy"] = out["sy"].diff().fillna(0.0)
    return out


def compute_step_label(df: pd.DataFrame, step_min=0.1, step_max=2.0) -> pd.Series:
    """
    这里仍是启发式 α 标签。你如果论文里有明确 α 公式，可替换这里。
    """
    v = df["vavg"].to_numpy(dtype=np.float64)
    vmin, vmax = float(np.min(v)), float(np.max(v))
    vnorm = np.zeros_like(v) if vmax - vmin < 1e-9 else (v - vmin) / (vmax - vmin)
    alpha = (1.0 - vnorm) * step_max + step_min
    alpha = np.clip(alpha, step_min, step_max)
    return pd.Series(alpha.astype(np.float32))


# -----------------------------
# 归一化
# -----------------------------
@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray


def compute_norm_stats(X: np.ndarray) -> NormStats:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return NormStats(mean=mean, std=std)


def apply_norm(X: np.ndarray, stats: NormStats) -> np.ndarray:
    return (X - stats.mean) / stats.std


# -----------------------------
# Dataset：y_gt 先只放 [dirx,diry,alpha]，c 由 MC 生成
# -----------------------------
class AlignDatasetA(Dataset):
    def __init__(self, X: np.ndarray, dir_gt: np.ndarray, alpha_gt: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.dir = torch.from_numpy(dir_gt.astype(np.float32))
        self.alpha = torch.from_numpy(alpha_gt.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.dir[idx], self.alpha[idx]


class AlignDatasetB(Dataset):
    def __init__(self, X: np.ndarray, c_target: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.c = torch.from_numpy(c_target.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.c[idx]


# -----------------------------
# 模型：输出 y=(B,4) 但训练分阶段
# -----------------------------
class AlignNetY4_MC(nn.Module):
    def __init__(self, in_dim=5, hidden=32, dropout_p=0.15):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(p=dropout_p)

        self.head_dir = nn.Linear(hidden, 2)
        self.head_alpha = nn.Linear(hidden, 1)
        self.head_conf = nn.Linear(hidden, 1)

    def trunk(self, x):
        h = torch.relu(self.fc1(x))
        h = self.drop(h)          # ★用于 MC 置信度
        h = torch.relu(self.fc2(h))
        h = self.drop(h)          # ★用于 MC 置信度
        return h

    def forward_dir_alpha(self, x):
        """
        只产生方向与步长（用于 Stage-A 训练、以及 Stage-B 计算 MC 置信度）
        返回：
          dir_unit (B,2), alpha (B,1)
        """
        h = self.trunk(x)

        dir_raw = self.head_dir(h)
        eps = 1e-8
        norm = torch.sqrt((dir_raw * dir_raw).sum(dim=-1, keepdim=True) + eps)
        dir_unit = dir_raw / norm

        alpha = torch.relu(self.head_alpha(h))  # >=0（Cube.AI 友好）

        return dir_unit, alpha

    def forward(self, x):
        """
        最终输出 y = [dirx, diry, alpha, c]
        注意：部署时 dropout 关闭（model.eval()）
        """
        dir_unit, alpha = self.forward_dir_alpha(x)
        # c 头：训练时回归 MC 计算得到的 C_target
        c = torch.sigmoid(self.head_conf(self.trunk(x)))  # (B,1)
        y = torch.cat([dir_unit, alpha, c], dim=-1)       # (B,4)
        return y


# -----------------------------
# Loss
# -----------------------------
def loss_stage_a(dir_pred, alpha_pred, dir_gt, alpha_gt, w_dir=1.0, w_alpha=0.3):
    # 方向：cosine loss
    cos = (dir_pred * dir_gt).sum(dim=-1).clamp(-1.0, 1.0)
    l_dir = (1.0 - cos).mean()
    l_alpha = F.mse_loss(alpha_pred.squeeze(-1), alpha_gt)
    return w_dir * l_dir + w_alpha * l_alpha


def loss_stage_b(c_pred, c_target):
    return F.mse_loss(c_pred.squeeze(-1), c_target)


# -----------------------------
# 新置信度定义：MC 一致性
# -----------------------------
@torch.no_grad()
def mc_confidence_C(model: AlignNetY4_MC,
                    x: torch.Tensor,
                    M: int = 16,
                    noise_std: float = 0.02,
                    noise_clip: float = 0.06) -> torch.Tensor:
    """
    C = || mean_j( u_tilde_j ) ||_2
    返回：(B,) in [0,1]
    """
    model.train()  # 让 dropout 生效
    u_list = []
    for _ in range(M):
        n = torch.randn_like(x) * noise_std
        n = torch.clamp(n, -noise_clip, noise_clip)
        x_pert = x + n
        dir_unit, _alpha = model.forward_dir_alpha(x_pert)  # (B,2)
        eps = 1e-8
        norm = torch.sqrt((dir_unit * dir_unit).sum(dim=-1, keepdim=True) + eps)
        u_tilde = dir_unit / norm
        u_list.append(u_tilde)

    U = torch.stack(u_list, dim=0)           # (M,B,2)
    mean_u = U.mean(dim=0)                   # (B,2)
    C = torch.sqrt((mean_u * mean_u).sum(dim=-1) + 1e-8)  # (B,)
    C = torch.clamp(C, 0.0, 1.0)
    return C


# -----------------------------
# Export: opset=13 + ir_version=9 (Cube.AI 10.2)
# -----------------------------
def export_onnx_ir9(model: nn.Module, out_onnx: str):
    import onnx

    model.eval()
    dummy = torch.zeros((1, 5), dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        out_onnx,
        input_names=["x"],
        output_names=["y"],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )

    m = onnx.load(out_onnx)
    m.ir_version = 9
    for ops in m.opset_import:
        if ops.domain in ("", "ai.onnx") and ops.version > 13:
            ops.version = 13
    onnx.save(m, out_onnx)


# -----------------------------
# Train pipeline (2-stage)
# -----------------------------
def train_and_export(sim_csv_path: str,
                     exp_csv_path: str,
                     out_onnx: str,
                     epochs_a: int = 60,
                     epochs_b: int = 30,
                     batch_size: int = 64,
                     lr_a: float = 1e-3,
                     lr_b: float = 5e-4,
                     hidden: int = 32,
                     dropout_p: float = 0.15,
                     seed: int = 42,
                     M_mc: int = 16,
                     noise_std: float = 0.02,
                     noise_clip: float = 0.06):

    np.random.seed(seed)
    torch.manual_seed(seed)

    # 数据准备：用【仿真数据】训练
    df_raw = parse_sim_csv(sim_csv_path)
    print("[SIM] parsed rows:", len(df_raw))
    grid = build_grid(df_raw)
    print("[SIM] grid points:", len(grid))

    grid_g = finite_difference_gradient(grid, field_col="vavg")
    data = compute_features(grid_g)

    # 标签：dir_gt 来自有限差分梯度；alpha_gt 启发式
    dir_gt = data[["gx", "gy"]].to_numpy(dtype=np.float32)
    dir_gt = dir_gt / (np.linalg.norm(dir_gt, axis=1, keepdims=True) + 1e-12)
    alpha_gt = compute_step_label(data).to_numpy(dtype=np.float32)

    # 输入特征
    X = data[["sx", "sy", "vavg", "dsx", "dsy"]].to_numpy(dtype=np.float32)

    ok = np.isfinite(X).all(axis=1) & np.isfinite(dir_gt).all(axis=1) & np.isfinite(alpha_gt)
    X, dir_gt, alpha_gt = X[ok], dir_gt[ok], alpha_gt[ok]
    if len(X) < 50:
        raise RuntimeError(f"有效样本太少：{len(X)}")

    # train/val split
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    n_train = int(0.9 * len(X))
    tr, va = idx[:n_train], idx[n_train:]

    # 归一化（基于仿真训练集）
    stats = compute_norm_stats(X[tr])
    Xn = apply_norm(X, stats).astype(np.float32)
    np.savez(out_onnx + ".norm.npz",
             mean=stats.mean.astype(np.float32),
             std=stats.std.astype(np.float32))

    # ============= 零点偏移校准：用【实验数据】估计 offset，并保存 =============
    # 对齐仿真与实验在特征空间的均值（只校准 sx, sy, vavg 这3个“零点”）
    try:
        df_exp_raw = parse_serial_debug(exp_csv_path)
        grid_exp = build_grid(df_exp_raw)
        data_exp = compute_features(grid_exp)

        X_exp_raw = data_exp[["sx","sy","vavg","dsx","dsy"]].to_numpy(dtype=np.float32)
        X_sim_raw = data[["sx","sy","vavg","dsx","dsy"]].to_numpy(dtype=np.float32)

        sim_mean = np.nanmean(X_sim_raw[:, :3], axis=0)
        exp_mean = np.nanmean(X_exp_raw[:, :3], axis=0)
        offset3 = (exp_mean - sim_mean).astype(np.float32)
        offset5 = np.array([offset3[0], offset3[1], offset3[2], 0.0, 0.0], dtype=np.float32)

        np.savez(out_onnx + ".offset.npz",
                 offset=offset5,
                 offset_sx=offset3[0],
                 offset_sy=offset3[1],
                 offset_vavg=offset3[2],
                 method="mean_align_sim",
                 sim_mean_3=sim_mean.astype(np.float32),
                 exp_mean_3=exp_mean.astype(np.float32))

        print("\n[CALIB] 已保存零点偏移:", out_onnx + ".offset.npz")
        print("[CALIB] offset[sx,sy,vavg] =", offset3)
    except Exception as e:
        print("\n[CALIB] 实验数据零点校准失败（将继续训练/导出，不影响模型生成）:")
        print("        ", repr(e))

    # ============= Stage-A: train dir + alpha =============
    ds_tr_a = AlignDatasetA(Xn[tr], dir_gt[tr], alpha_gt[tr])
    ds_va_a = AlignDatasetA(Xn[va], dir_gt[va], alpha_gt[va])
    dl_tr_a = DataLoader(ds_tr_a, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va_a = DataLoader(ds_va_a, batch_size=batch_size, shuffle=False, drop_last=False)

    model = AlignNetY4_MC(in_dim=5, hidden=hidden, dropout_p=dropout_p)
    opt_a = torch.optim.Adam(model.parameters(), lr=lr_a)

    print("\n[Stage-A] train direction + alpha ...")
    for ep in range(1, epochs_a + 1):
        model.train()
        tr_loss = 0.0
        for xb, dgb, agb in dl_tr_a:
            dir_pred, alpha_pred = model.forward_dir_alpha(xb)
            loss = loss_stage_a(dir_pred, alpha_pred, dgb, agb)
            opt_a.zero_grad()
            loss.backward()
            opt_a.step()
            tr_loss += float(loss.item()) * xb.size(0)
        tr_loss /= max(1, len(ds_tr_a))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, dgb, agb in dl_va_a:
                dir_pred, alpha_pred = model.forward_dir_alpha(xb)
                loss = loss_stage_a(dir_pred, alpha_pred, dgb, agb)
                va_loss += float(loss.item()) * xb.size(0)
        va_loss /= max(1, len(ds_va_a))

        if ep == 1 or ep % 10 == 0 or ep == epochs_a:
            print(f"EpochA {ep:3d} | train={tr_loss:.6f} | val={va_loss:.6f}")

    # ============= Stage-B: build MC confidence targets, then train c_head =============
    print("\n[Stage-B] build MC confidence targets (C) ...")
    X_tensor = torch.from_numpy(Xn).float()
    C_all = []
    bs_mc = 256
    for i in range(0, X_tensor.size(0), bs_mc):
        xb = X_tensor[i:i + bs_mc]
        Cb = mc_confidence_C(model, xb, M=M_mc, noise_std=noise_std, noise_clip=noise_clip)
        C_all.append(Cb.cpu().numpy())
    C_all = np.concatenate(C_all, axis=0).astype(np.float32)

    # 训练 c_head：冻结 trunk/dir/alpha，只训练 head_conf
    for p in model.fc1.parameters(): p.requires_grad = False
    for p in model.fc2.parameters(): p.requires_grad = False
    for p in model.head_dir.parameters(): p.requires_grad = False
    for p in model.head_alpha.parameters(): p.requires_grad = False
    for p in model.head_conf.parameters(): p.requires_grad = True

    ds_tr_b = AlignDatasetB(Xn[tr], C_all[tr])
    ds_va_b = AlignDatasetB(Xn[va], C_all[va])
    dl_tr_b = DataLoader(ds_tr_b, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va_b = DataLoader(ds_va_b, batch_size=batch_size, shuffle=False, drop_last=False)

    opt_b = torch.optim.Adam(model.head_conf.parameters(), lr=lr_b)

    print("[Stage-B] train confidence head ...")
    for ep in range(1, epochs_b + 1):
        model.train()
        tr_loss = 0.0
        for xb, cb in dl_tr_b:
            h = model.trunk(xb)
            c_pred = torch.sigmoid(model.head_conf(h))
            loss = loss_stage_b(c_pred, cb)
            opt_b.zero_grad()
            loss.backward()
            opt_b.step()
            tr_loss += float(loss.item()) * xb.size(0)
        tr_loss /= max(1, len(ds_tr_b))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, cb in dl_va_b:
                h = model.trunk(xb)
                c_pred = torch.sigmoid(model.head_conf(h))
                loss = loss_stage_b(c_pred, cb)
                va_loss += float(loss.item()) * xb.size(0)
        va_loss /= max(1, len(ds_va_b))

        if ep == 1 or ep % 5 == 0 or ep == epochs_b:
            print(f"EpochB {ep:3d} | train={tr_loss:.6f} | val={va_loss:.6f}")

    export_onnx_ir9(model, out_onnx)
    print(f"\nExported for Cube.AI: {out_onnx}")
    print(f"Saved norm stats: {out_onnx}.norm.npz")
    print("输出 y(1,4)=[dir_x, dir_y, alpha, conf]")
    print("注意：MCU 端推理前必须对输入做 (x-mean)/std 归一化。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=str, default=None, help="仿真数据CSV")
    ap.add_argument("--exp", type=str, default=None, help="实验数据CSV(用于零点校准)")
    ap.add_argument("--out", type=str, default=OUT_ONNX_DEFAULT)

    ap.add_argument("--epochs_a", type=int, default=60)
    ap.add_argument("--epochs_b", type=int, default=30)

    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr_a", type=float, default=1e-3)
    ap.add_argument("--lr_b", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.15)

    ap.add_argument("--M", type=int, default=16, help="MC repeats for confidence")
    ap.add_argument("--noise_std", type=float, default=0.02)
    ap.add_argument("--noise_clip", type=float, default=0.06)

    args = ap.parse_args()

    sim_csv = args.sim if args.sim else SIM_CSV_DEFAULT
    exp_csv = args.exp if args.exp else EXP_CSV_DEFAULT
    if not sim_csv or not os.path.exists(sim_csv):
        raise FileNotFoundError(
            "没有找到仿真 CSV。\n"
            "方法1：python 脚本.py --sim R=140三维.csv --exp sss.csv\n"
            f"方法2：修改脚本顶部 SIM_CSV_DEFAULT（当前：{SIM_CSV_DEFAULT}）"
        )
    if not exp_csv or not os.path.exists(exp_csv):
        print("[WARN] 没有找到实验CSV，将跳过零点校准：", exp_csv)

    train_and_export(
        sim_csv_path=sim_csv,
        exp_csv_path=exp_csv,
        out_onnx=args.out,
        epochs_a=args.epochs_a,
        epochs_b=args.epochs_b,
        batch_size=args.batch,
        lr_a=args.lr_a,
        lr_b=args.lr_b,
        hidden=args.hidden,
        dropout_p=args.dropout,
        M_mc=args.M,
        noise_std=args.noise_std,
        noise_clip=args.noise_clip,
    )
    d = np.load(args.out + ".norm.npz")
    print("MEAN =", d["mean"])
    print("STD  =", d["std"])


if __name__ == "__main__":
    main()
