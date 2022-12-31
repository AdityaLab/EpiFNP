import subprocess
from runs.consts import regions

for curr in [2, 4, 5]:
    for region in regions:
        print(f"Training {region}...")
        subprocess.run(
            [
                "python",
                "train_ili_shift_max.py",
                "-y",
                str(2022),
                "-w",
                "2",
                "-a",
                "trans",
                "-r",
                region,
                "-e",
                str(5500),
                "-c",
                str(curr),
            ]
        )
