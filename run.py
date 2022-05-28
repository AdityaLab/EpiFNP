import subprocess
import numpy as np
import os

for test_years in [2014, 2015, 2016, 2017, 2018, 2019]:
    for week in [2, 3, 4]:
        for atten in [
            "trans",
        ]:
            model_name = f"epifnp_{test_years}_{week}"
            print(model_name)
            r = subprocess.call(
                [
                    "python",
                    "train_ili.py",
                    "-y",
                    str(test_years),
                    "-w",
                    str(week),
                    "-a",
                    atten,
                    "-n",
                    model_name,
                    "-e",
                    str(3000),
                ]
            )
            if r != 0:
                raise Exception(f"{model_name} process encountered error")
