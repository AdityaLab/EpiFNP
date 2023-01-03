import subprocess
from runs.consts import imp_states as states

for curr in range(11):
    for wt in [50, 500]:
        for region in states:
            print(f"Training {region}...")
            subprocess.run(
                [
                    "python",
                    "train_ili_max_weight.py",
                    "-y",
                    str(2022),
                    "-a",
                    "trans",
                    "-d",
                    "gru",
                    "-r",
                    region,
                    "-e",
                    str(5500),
                    "-c",
                    str(curr),
                    "--weight",
                    str(wt),
                ]
            )
