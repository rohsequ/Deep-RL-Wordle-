from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb

# major_ver, minor_ver, _ = version.parse(tb.__version__).release
# assert major_ver >= 2 and minor_ver >= 3, \
#     "This notebook requires TensorBoard 2.3 or later."
# print("TensorBoard version: ", tb.__version__)


experiment_id = "PBQhBAJ1TlukeCLAwSznNg"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()
# df.to_csv('5-letter-full-character-level.csv')

df_ver11 = df[df['run'] == 'version_11']
df_ver11 = df_ver11[df_ver11['tag']=='loss_ratio:']

steps_all = []
values_all = []

steps = df_ver11['step'].to_numpy()
value = df_ver11['value'].to_numpy()

steps_all.extend(list(steps))
values_all.extend(list(value))

df_ver12 = df[df['run'] == 'version_12']
df_ver12 = df_ver12[df_ver12['tag']=='loss_ratio:']

steps = df_ver12['step'].to_numpy()
value = df_ver12['value'].to_numpy()

steps += steps_all[-1]

steps_all.extend(list(steps))
values_all.extend(list(value))

df_ver13 = df[df['run'] == 'version_13']
df_ver13 = df_ver13[df_ver13['tag']=='loss_ratio:']

steps = df_ver13['step'].to_numpy()
value = df_ver13['value'].to_numpy()

steps += steps_all[-1]

steps_all.extend(list(steps))
values_all.extend(list(value))

df_ver14 = df[df['run'] == 'version_14']
df_ver14 = df_ver14[df_ver14['tag']=='loss_ratio:']

steps = df_ver14['step'].to_numpy()
value = df_ver14['value'].to_numpy()

steps += steps_all[-1]

steps_all.extend(list(steps))
values_all.extend(list(value))

df_out = pd.DataFrame()
df_out["Steps"] = steps_all
df_out["Loss Ratio"] = values_all

df_out.to_csv('5-letter-loss-ratios.csv')
plt.plot(steps_all, values_all)
plt.show()
