import subprocess
from runs.consts import imp_states as states
import itertools

states = ["X"]
for curr in range(11):
    for shift_hor, shift_vert, weight, lookback in itertools.product(
        [False], [False], [1, 2, 5], [0, 10, 20]
    ):
        for region in states:
            print(f"Training {region}...")
            command = [
                "python",
                "train_ili_max_instance_lookback.py",
                "-y",
                str(2022),
                "--shift_vert" if shift_vert else "",
                "--shift_hor" if shift_hor else "",
                "--weight",
                str(weight),
                "--lookback",
                str(lookback),
                "-r",
                region,
                "-c",
                str(curr),
            ]
            print("Running: ", " ".join(command))
            subprocess.run(command)
