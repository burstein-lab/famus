import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

log_file = "tmp/logs/2024_03_03_10_05_02_475004"

lines = open(log_file, "r").readlines()[::10]

pat = r"Moving average loss: 0\.\d+"

losses = [
    float(re.findall(pat, line)[0].split()[-1])
    for line in lines
    if re.findall(pat, line)
]

losses = np.log10(losses)

sns.lineplot(x=range(len(losses)), y=losses)

pat = r"Moving average eval loss: 0\.\d+"

losses = [
    float(re.findall(pat, line)[0].split()[-1])
    for line in lines
    if re.findall(pat, line)
]

losses = np.log10(losses)

sns.lineplot(x=range(len(losses)), y=losses)

plt.savefig("losses.png", dpi=300)
