import os
import random
import pickle
from pathlib import Path
from collections import defaultdict

def collect_ir_paths(base_dir: Path):
    """
    tx_rx単位でnpzファイルパスを収集し、辞書で返す。
    """
    tx_rx_dict = defaultdict(list)
    for tx_dir in sorted(base_dir.glob("tx_*")):
        for rx_dir in sorted(tx_dir.glob("rx_*")):
            key = f"{tx_dir.name}/{rx_dir.name}"
            npz_files = sorted(rx_dir.glob("*.npz"))
            tx_rx_dict[key].extend(npz_files)
    return tx_rx_dict

def split_train_test(tx_rx_dict, test_ratio=0.2, seed=42):
    """
    tx_rx単位でtrain/testに分割する。
    """
    random.seed(seed)
    keys = list(tx_rx_dict.keys())
    random.shuffle(keys)
    
    num_test = int(len(keys) * test_ratio)
    test_keys = set(keys[:num_test])
    train_keys = set(keys[num_test:])

    split = {
        "train": [str(p) for k in train_keys for p in tx_rx_dict[k]],
        "test": [str(p) for k in test_keys for p in tx_rx_dict[k]],
    }
    return split

def save_split(split, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(split, f)
    print(f"Saved split to: {output_path}")

# 実行例
if __name__ == "__main__":
    outputs_dir = Path("./outputs/real_exp_8720")  # 適宜パスを変更してください
    tx_rx_dict = collect_ir_paths(outputs_dir)
    split = split_train_test(tx_rx_dict, test_ratio=0.2)
    save_split(split, outputs_dir / "train_test_split.pkl")
