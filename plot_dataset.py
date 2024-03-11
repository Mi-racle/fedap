import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv('results/dataset_stats_dirichlet.csv', header=None)
df = pd.read_csv('results/dataset_stats_label.csv', header=None)
df = df.transpose()

df.plot(kind='bar', stacked=True, colormap='viridis', figsize=(8, 6))

plt.title('Data distribution')
plt.xlabel('type')
plt.ylabel('number')

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# plt.savefig('results/dirichlet_distribution.png')
plt.savefig('results/label_distribution.png')
