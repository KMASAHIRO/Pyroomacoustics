import os
import numpy as np
import pyroomacoustics as pra
from tqdm import tqdm
import json

def generate_positions_real_env():
    # 部屋隅が原点
    x_offset = 0.0 + 1.0  # X=1.0m から開始
    y_offset = 0.0 + 1.5  # Y=1.5m から開始
    z_pos = 0.0 + 1.5       # 床から1.5m上 = 0.15

    # グリッド全体の24箇所をリストアップ
    all_centers = []
    for i in range(6):  # Y方向（奥行）
        for j in range(4):  # X方向（幅）
            x = x_offset + j * 1.0
            y = y_offset + i * 1.0
            all_centers.append([x, y, z_pos])
    all_centers = np.array(all_centers)  # shape: (24, 3)

    # スピーカー配置：四隅 + 中央4点のインデックス
    spk_indices = [0*4+0, 0*4+3, 5*4+0, 5*4+3, 2*4+1, 2*4+2, 3*4+1, 3*4+2]
    tx_pos = all_centers[spk_indices]  # shape: (8, 3)

    # 定数定義
    num_channels = 8
    radius = 0.0365  # マイク円半径
    num_speakers = len(spk_indices)
    num_total = len(all_centers)
    num_mics_per_spk = num_total - 1  # 1つをスピーカーに使うので残りがマイク

    # 出力配列初期化
    rx_pos = np.zeros((num_speakers, num_mics_per_spk, num_channels, 3))
    mic_centers_per_spk = []

    for s_idx, spk_idx in enumerate(spk_indices):
        # スピーカー以外の23個のマイク中心位置を取得
        mic_indices = [i for i in range(num_total) if i != spk_idx]
        mic_centers = all_centers[mic_indices]  # shape: (23, 3)
        mic_centers_per_spk.append(mic_centers)

        for m_idx, (cx, cy, cz) in enumerate(mic_centers):
            for ch in range(num_channels):
                # Y+方向を1ch目として反時計回りに配置
                theta = np.pi / 2 + ch * (2 * np.pi / num_channels)
                x = cx + radius * np.cos(theta)
                y = cy + radius * np.sin(theta)
                rx_pos[s_idx, m_idx, ch] = [x, y, cz]

    return tx_pos, mic_centers_per_spk, rx_pos

def simulate_pyroomacoustics_ir(
    output_path,
    room_dim=(6.110, 8.807, 2.7),
    sampling_rate=48000,
    max_order=10,
    e_absorption=0.0055,
    mic_num=8,
    ir_len=4800
):
    # スピーカーとマイク位置（円形配置）を生成
    tx_all, mic_centers_all, rx_all = generate_positions_real_env()

    # スピーカー情報を保存
    speaker_data = {
        "speaker": {
            "positions": tx_all.tolist()
        }
    }
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'speaker_data.json'), 'w') as f:
        json.dump(speaker_data, f, indent=4)

    for tx_index, tx_pos in tqdm(enumerate(tx_all), total=len(tx_all), desc="Pyroom IR Sim"):
        tx_output_path = os.path.join(output_path, f"tx_{tx_index}")
        os.makedirs(tx_output_path, exist_ok=True)

        # Room構築
        room = pra.ShoeBox(
            room_dim,
            fs=sampling_rate,
            materials=pra.Material(e_absorption),
            max_order=max_order
        )
        room.add_source(tx_pos.tolist())

        # generate_positionsで生成済みの円形マイク位置を使用（shape: (23, mic_num, 3)）
        rx_pos = rx_all[tx_index]              # shape: (23, mic_num, 3)
        mic_positions = rx_pos.reshape(-1, 3).T  # shape: (3, 23×mic_num)

        room.add_microphone_array(mic_positions)
        room.compute_rir()

        # IRを ir_len サンプルにクリップして保存
        for i, ir in enumerate(room.rir):
            ir_clipped = np.array(ir[0][:ir_len])
            np.savez(
                os.path.join(tx_output_path, f'ir_{str(i).zfill(6)}.npz'),
                ir=ir_clipped,
                position_rx=mic_positions[:, i],
                position_tx=np.array(tx_pos)
            )

if __name__ == "__main__":
    simulate_pyroomacoustics_ir(output_path="./outputs/real_env_avr")
