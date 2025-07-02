import subprocess
import numpy as np
import random


np.random.seed(42)

base_prototypes = [2, 3, 2, 3, 1, 1, 1, 3, 4, 1,
                   1, 2, 3, 4, 1, 4, 3, 1, 1, 2]

prototype_options = [
    [max(1, b-1), b, min(5, b+1)]
    for b in base_prototypes
]

N_SAMPLES = 50
sampled_combinations = []

for _ in range(N_SAMPLES):
    config = [random.choice(opts) for opts in prototype_options]
    sampled_combinations.append(config)

print(f"Sampled {len(sampled_combinations)} configurations.")

for i, config in enumerate(sampled_combinations):
    config_str = "[" + ", ".join(str(x) for x in config) + "]"

    # Compose the command
    cmd = [
        "python", "main.py",
        "training_contrastive",
        "-c", "/noc/users/mojmas/files/code/AMP/configs/config_cifar-3.yaml",
        "-i", "/noc/users/mojmas/files/data/UVP6Net/",
        "-o", f"/noc/users/mojmas/files/data/",
        "-p", config_str
    ]

    print(f"\n=== Running config {i+1}/{len(sampled_combinations)} ===")
    print("Command:", " ".join(cmd))

    subprocess.run(cmd)

