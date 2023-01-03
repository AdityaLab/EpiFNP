import subprocess
from runs.consts import states as states

for curr in range(11):
    for decoder in ["gru"]:
        for region in states:
            print(f"Training {region}...")
            subprocess.run(
                [
                    "python",
                    "train_ili_max_ar.py",
                    "-y",
                    str(2022),
                    "-a",
                    "trans",
                    "-d",
                    decoder,
                    "-r",
                    region,
                    "-e",
                    str(5500),
                    "-c",
                    str(curr),
                ]
            )
