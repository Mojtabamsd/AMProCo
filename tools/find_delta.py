import subprocess
import ast
import os

input_path = os.path.join(os.path.dirname(__file__), "prediction.txt")
output_path = os.path.join(os.path.dirname(__file__), "prediction_updated.txt")

with open(input_path, "r") as f:
    lines = f.readlines()

with open(output_path, "w") as f_out:
    for i, line in enumerate(lines):
        try:
            # Extract config list
            config_part = line.strip().split("],")[0] + "]"
            config_list = ast.literal_eval(config_part)
            config_str = str(config_list)

            # Compose command
            cmd = [
                "python", "main.py",
                "training_contrastive",
                "-c", "/noc/users/mojmas/files/code/AMP/configs/config_cifar-2.yaml",
                "-i", "/noc/users/mojmas/files/data/UVP6Net/",
                "-o", "/noc/users/mojmas/files/data/",
                "-p", config_str
            ]

            print(f"\n=== Running config {i+1}/{len(lines)} ===")
            print("Command:", " ".join(cmd))

            # Run the command and capture output
            # Run the command and capture output
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout.strip()

            # Try to find line like: best global δ ≈ 6587.0
            delta_str = "best global delta:"
            value = "NaN"  # fallback

            for line in output.splitlines():
                if delta_str in line:
                    try:
                        value = float(line.split(":")[1].strip())
                    except Exception:
                        value = "NaN"
                    break

            new_line = lines.strip() + f", {value}\n"
            f_out.write(new_line)

        except Exception as e:
            print(f"Error on line {i+1}: {e}")
            f_out.write(line.strip() + ", ERROR\n")