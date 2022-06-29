import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["font.size"] = 12

alessio = [26, 131, 27, 40, 123, 4]
mine = [29, 160, 56, 48, 74, 2]
rec_names = ["2950", "2953", "2954", "2957", "5116", "5118"]
df = pd.DataFrame({"Previous API's": alessio,
                   "Max Lim's": mine},
                  index=rec_names)
df.plot.bar(rot=0, color=["#8b0717", "#0000FF"])
plt.xlabel("Multi-Electrode Array ID")
plt.ylabel("Number of units")
plt.legend(loc="upper right")
plt.tick_params(axis="x", colors="red")
plt.show()
