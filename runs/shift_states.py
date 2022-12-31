import subprocess
from runs.consts import states as states

for curr in [8]:
    for ahead in [4]:
        for region in states:
            print(f"Training {region}...")
            subprocess.run(
                [
                    "python",
                    "train_ili_max_states.py",
                    "-y",
                    str(2022),
                    "-w",
                    str(ahead),
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
