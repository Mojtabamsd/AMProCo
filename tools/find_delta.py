import subprocess
import ast
import os

file_path = os.path.join(os.path.dirname(__file__), "prediction.txt")

with open(file_path, "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    try:
        config_part = line.strip().split("],")[0] + "]"
        config_list = ast.literal_eval(config_part)
        config_str = str(config_list)

        # Compose the command
        cmd = [
            "python", "main.py",
            "training_contrastive",
            "-c", "/noc/users/mojmas/files/code/AMP/configs/config_cifar-3.yaml",
            "-i", "/noc/users/mojmas/files/data/UVP6Net/",
            "-o", "/noc/users/mojmas/files/data/",
            "-p", config_str
        ]

        print(f"\n=== Running config {i+1}/{len(lines)} ===")
        print("Command:", " ".join(cmd))

        subprocess.run(cmd)
    except Exception as e:
        print(f"Skipping line {i+1} due to error: {e}")
