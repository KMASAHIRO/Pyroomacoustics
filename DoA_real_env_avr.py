import os
import numpy as np
import pyroomacoustics as pra
import math
import pickle

def load_ir_npz(file_path):
    data = np.load(file_path)
    return data['ir'], data['position_rx'], data['position_tx']

def estimate_doa_for_algorithms(
    rx_folder,
    fs=48000,
    n_fft=512,
    mic_radius=0.0365,
    algo_names=None
):
    if algo_names is None:
        algo_names = ['MUSIC', 'NormMUSIC', 'SRP', 'CSSM', 'WAVES', 'TOPS', 'FRIDA']

    ir_files = sorted([f for f in os.listdir(rx_folder) if f.endswith('.npz')])
    mic_num = len(ir_files)

    ir_data = [load_ir_npz(os.path.join(rx_folder, f)) for f in ir_files]
    ir_signals = [d[0] for d in ir_data]
    rx_positions = [d[1] for d in ir_data]
    tx_position = ir_data[0][2]

    signals = np.stack(ir_signals, axis=0)
    mic_positions = np.stack(rx_positions, axis=1)  # shape: (3, mic_num)

    mic_center = np.mean(mic_positions[:2, :], axis=1)
    mic_geometry = pra.beamforming.circular_2D_array(
        center=mic_center, M=mic_num, radius=mic_radius, phi0=np.pi / 2
    )

    X = np.array([
        pra.transform.stft.analysis(sig, n_fft, n_fft // 2)
        for sig in signals
    ])
    X = np.transpose(X, (0, 2, 1))  # (M, F, S)

    dx, dy = tx_position[0] - mic_center[0], tx_position[1] - mic_center[1]
    true_rad = math.atan2(dy, dx)
    if true_rad < 0:
        true_rad += 2 * np.pi
    true_deg = np.degrees(true_rad)

    result_per_algo = {}
    for algo in algo_names:
        doa = pra.doa.algorithms[algo](mic_geometry, fs=fs, nfft=n_fft)
        doa.locate_sources(X)  # freq_range はデフォルト [500, 4000]

        if algo == 'FRIDA':
            doa_values = doa._gen_dirty_img()
            doa_degree = np.argmax(np.abs(doa_values))
        else:
            doa_values = doa.grid.values
            doa_degree = np.argmax(doa_values)

        error = min(abs(doa_degree - true_deg), 360 - abs(doa_degree - true_deg))

        result_per_algo[algo] = {
            "true_deg": true_deg,
            "est_deg": doa_degree,
            "error": error,
            "doa_values": doa_values
        }

    return result_per_algo

def run_doa_all_tx(
    base_dir="./outputs/real_env_avr",
    fs=48000,
    n_fft=512,
    mic_radius=0.0365,
    algo_names=None,
    save_filename="DoA.pkl"
):
    if algo_names is None:
        algo_names = ['MUSIC', 'NormMUSIC', 'SRP', 'CSSM', 'WAVES', 'TOPS', 'FRIDA']

    tx_folders = sorted([f for f in os.listdir(base_dir) if f.startswith("tx_")])
    results = {algo: {"source_direction": {}, "DoA": {}} for algo in algo_names}
    error_stats = {algo: [] for algo in algo_names}

    for tx_name in tx_folders:
        tx_path = os.path.join(base_dir, tx_name)
        rx_folders = sorted([f for f in os.listdir(tx_path) if f.startswith("rx_")])

        for rx_name in rx_folders:
            rx_path = os.path.join(tx_path, rx_name)
            key = f"{tx_name}_{rx_name}"

            result_per_algo = estimate_doa_for_algorithms(
                rx_folder=rx_path,
                fs=fs,
                n_fft=n_fft,
                mic_radius=mic_radius,
                algo_names=algo_names
            )

            for algo in algo_names:
                res = result_per_algo[algo]
                results[algo]["source_direction"][key] = res["true_deg"]
                results[algo]["DoA"][key] = res["doa_values"]
                error_stats[algo].append(res["error"])
                #print(f"[{algo}] {key} → True: {res['true_deg']:.2f}°, Est: {res['est_deg']:.2f}°, Error: {res['error']:.2f}°")

    print("\n=== DoA Estimation Summary ===")
    for algo in algo_names:
        errors = np.array(error_stats[algo])
        print(f"{algo:10s} → Mean Error: {errors.mean():.2f}°, Std: {errors.std():.2f}°")

    with open(os.path.join(base_dir, save_filename), "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    run_doa_all_tx()
