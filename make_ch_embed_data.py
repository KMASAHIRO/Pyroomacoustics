import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

def process_rx_directory(rx_dir: Path, save_dir: Path):
    ir_files = sorted(rx_dir.glob("ir_*.npz"))
    if len(ir_files) == 0:
        print(f"Warning: No IR files in {rx_dir}. Skipping.")
        return

    # 全IRファイルを読み込む
    position_rx_all = []
    ir_data_all = []

    for f in ir_files:
        data = np.load(f)
        position_rx_all.append(data["position_rx"])
        ir_data_all.append({
            "ir": data["ir"],
            "position_tx": data["position_tx"],
        })

    position_rx_all = np.array(position_rx_all)
    center = position_rx_all.mean(axis=0)  # (3,)

    os.makedirs(save_dir, exist_ok=True)

    # 全チャネルに対して保存処理（Y+から反時計回りにch_idxを振る）
    for ch_idx, f_data in enumerate(ir_data_all):
        save_path = save_dir / f"ir_{str(ch_idx).zfill(6)}.npz"
        np.savez(
            save_path,
            ir=f_data["ir"],
            position_rx=center,  # 中心に統一
            position_tx=f_data["position_tx"],
            ch_idx=ch_idx
        )

def process_all_rx_dirs(input_root: Path, output_root: Path):
    for tx_dir in sorted(input_root.glob("tx_*")):
        tx_index = tx_dir.name.split("_")[-1]
        for rx_dir in sorted(tx_dir.glob("rx_*")):
            rx_index = rx_dir.name.split("_")[-1]
            output_rx_dir = output_root / f"tx_{tx_index}" / f"rx_{rx_index}"
            process_rx_directory(rx_dir, output_rx_dir)

# 使用例
input_root = Path("./outputs/real_env_avr_16kHz")
output_root = Path("./outputs/real_env_avr_16kHz_centered")

process_all_rx_dirs(input_root, output_root)
