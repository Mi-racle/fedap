import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

CSV_PATHS = [
    'results/central/confusion_maxtrix_central.csv',
    'results/dirichlet-fedap/confusion_matrix_cosine.csv',
    'results/dirichlet-fedavg/confusion_matrix_avg.csv',
    'results/dirichlet-fedprox/confusion_matrix_prox.csv',
    'results/iid-fedacc/confusion_matrix_acc.csv',
    'results/iid-fedavg/confusion_matrix_avg.csv',
    'results/label-fedap/confusion_matrix_cosine.csv',
    'results/label-fedavg/confusion_matrix_avg.csv',
    'results/label-fedprox/confusion_matrix_prox.csv',
]

OUTPUT_DIRS = [
    'results/central/',
    'results/dirichlet-fedap/',
    'results/dirichlet-fedavg/',
    'results/dirichlet-fedprox/',
    'results/iid-fedacc/',
    'results/iid-fedavg/',
    'results/label-fedap/',
    'results/label-fedavg/',
    'results/label-fedprox/',
]

P_R_F1_NAME = 'p_r_f1.csv'
CM_NAME = 'confusion_matrix.png'

if __name__ == '__main__':

    for i, csv_path in enumerate(CSV_PATHS):

        p_r_f1_path = OUTPUT_DIRS[i] + P_R_F1_NAME
        cm_path = OUTPUT_DIRS[i] + CM_NAME

        matrix_data = pd.read_csv(csv_path, header=None)
        confusion_matrix = matrix_data.values

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix * .2, fmt='f', cmap='Blues', xticklabels=range(53), yticklabels=range(53))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.savefig(cm_path, bbox_inches='tight')

        true_positive = np.diagonal(confusion_matrix)
        true = np.sum(confusion_matrix, axis=1)
        predicted = np.sum(confusion_matrix, axis=0)

        precision = true_positive / true
        recall = true_positive / predicted
        f1_score = 2 * precision * recall / (precision + recall)
        f1_score = np.nan_to_num(f1_score)

        total_samples = np.sum(true)
        total_precision = np.sum(precision * true) / total_samples
        total_recall = np.sum(recall * true) / total_samples
        total_f1_score = np.sum(f1_score * true) / total_samples
        totals = np.array([total_precision, total_recall, total_f1_score])

        output_table = np.stack([precision, recall, f1_score])
        output_table = np.round(output_table, decimals=4)
        output_table = np.transpose(output_table)
        output_table = np.vstack([output_table, totals])

        np.savetxt(p_r_f1_path, output_table, fmt='%.4f', delimiter=',')
