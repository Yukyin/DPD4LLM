import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import visualization

class DataVisualizer:
    def __init__(self):
        pass

    def load_data(self, file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)

    def visualize_text_data(self, data):
        visualization.visualize_text(data)

    def plot_heatmap(self, matrix, figsize=(10, 10), cmap='YlGnBu', xlabel='Head (sorted)', ylabel='Layer'):
        plt.figure(figsize=figsize)
        sns.heatmap(matrix, cmap=cmap)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_correlation_matrix(self, correlation_matrix, figsize=(10, 10), cmap='coolwarm', xticklabels=range(0,9), yticklabels=range(0,9), save_path=None):
        plt.figure(figsize=figsize)
        sns.heatmap(correlation_matrix, annot=True, cmap=cmap, xticklabels=xticklabels, yticklabels=yticklabels)
        print(correlation_matrix.min())
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=1000, bbox_inches='tight')
        plt.show()

    def plot_scatter(self, matrix_pca, labels, count, xlim=(-300, 300), ylim=(-100, 100)):
        for i in range(2):  # Assuming binary labels: 0 and 1
            label = 'False (Type {})'.format(count) if i == 1 and count != 9 else 'True' if i == 0 else 'False'
            plt.scatter(matrix_pca[labels == i, 0], matrix_pca[labels == i, 1], label=label)
        plt.legend()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()

    def plot_line(self, x, y1, y2, count, xticks=range(0, 351, 50), xticklabels=['q', 'k', 'v', 'o', 'gate', 'up', 'down', 'emb']):
        plt.plot(x, y1, label='True')
        label = 'False' if count == 9 else 'False (Type {})'.format(count)
        plt.plot(x, y2, label=label)
        plt.xticks(xticks, xticklabels)
        plt.legend()
        plt.show()

    def plot_overall_line(self, data_to_load):
        for data in data_to_load:
            x, y1, y2, count = data
            if count == 9:
                break
            self.plot_line(x, y1, y2, count)
 
visualizer = DataVisualizer()
 
for i in range(9):
    vis_data_records_ig = visualizer.load_data('vis_data_records_ig{}.pkl'.format(i))
    visualizer.visualize_text_data(vis_data_records_ig)
 
matrix_list = visualizer.load_data('heatmap_variables-3.pkl')
for matrix in matrix_list:
    visualizer.plot_heatmap(matrix)
 
correlation_matrix = visualizer.load_data('variables-3.pkl')
visualizer.plot_correlation_matrix(correlation_matrix, save_path='correlationfig.pdf')
 
data_to_load = visualizer.load_data('matrix_pca_data_to_save-3.pkl')
for matrix_pca, labels, count in data_to_load:
    visualizer.plot_scatter(matrix_pca, labels, count)
 
data_to_load = visualizer.load_data('line_data_to_save-3.pkl')
for data in data_to_load:
    x, y1, y2, count = data
    visualizer.plot_line(x, y1, y2, count)
 
visualizer.plot_overall_line(data_to_load)
