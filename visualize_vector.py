# import plotly.offline as py
import sys
import codecs
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle as pk
import numpy as np
from pandas import read_csv
from sklearn.cluster import KMeans
file_to_save_vector = 'results/mem/5minutes/vector_presentation24-5-3-32-2-1-1-1-1-8-1.csv'
def read_trained_data(file_trained_data):
    vector_df = read_csv(file_trained_data, header=None, index_col=False, engine='python')
    vector = vector_df.values
    return vector 
vectors = read_trained_data(file_to_save_vector)

number_of_vecs = len(vectors)

all_vec = []
for i in range(number_of_vecs):
    # print (vectors[i])
    all_vec.append(i)

def main():
    # embeddings_file = sys.argv[1]
    # wv, vocabulary = load_embeddings(embeddings_file)
 
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(vectors[:number_of_vecs,:])
    kmeans = KMeans(n_clusters=12)
    kmeans.fit(Y)   
    plt.scatter(Y[:, 0], Y[:, 1],c=kmeans.labels_, cmap='rainbow')
    plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black') 
    for label, x, y in zip(all_vec, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()
 

if __name__ == '__main__':
    main()