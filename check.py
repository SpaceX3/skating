import os
import numpy as np


def list_npy(dir_path):
    if not os.path.isdir(dir_path):
        return {}
    files = [f for f in os.listdir(dir_path) if f.endswith(".npy")]
    return {os.path.splitext(f)[0]: os.path.join(dir_path, f) for f in files}


def main():
    # 两个待对比的目录（与 dataset_fs800.py 一致）
    dir_new = os.path.join("..", "FS1000 Dataset", "VST_feature_fs800")
    dir_old = os.path.join("..", "FS1000 Dataset", "Timesformer_output_feature_fs800")

    map_new = list_npy(dir_new)
    map_old = list_npy(dir_old)

    common_keys = sorted(set(map_new.keys()) & set(map_old.keys()))
    if not common_keys:
        print("No common files found between:")
        print(" -", dir_new)
        print(" -", dir_old)
        return

    print("Compare feature shapes for up to 10 common files:")
    count = 0
    for key in common_keys:
        try:
            a = np.load(map_new[key], mmap_mode="r")
            b = np.load(map_old[key], mmap_mode="r")
            print(f"{key}: new(VST) {a.shape} | old(TimeSformer) {b.shape}")
        except Exception as e:
            print(f"{key}: error -> {e}")
        count += 1
        if count >= 10:
            break


if __name__ == "__main__":
    main()

