import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 2d visualization of vectors
def viz_2d(vectors, categories):
    fig = plt.figure()
    ax = fig.add_subplot()

    xs = [element[0] for element in vectors]
    ys = [element[1] for element in vectors]

    ax.scatter(xs, ys, c=categories)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("2-D representation of vectors")

    return fig


# 3d visualization of vectors
def viz_3d(vectors, categories):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xs = [element[0] for element in vectors]
    ys = [element[1] for element in vectors]
    zs = [element[2] for element in vectors]

    ax.scatter(xs, ys, zs, c=categories)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_label('Z')
    ax.set_title("3-D representation of vectors")

    return fig

# 2-d reduction of n-dimensional vectors

def umap(vectors):
    reducer = umap.UMAP()
    scaled_vectors = StandardScaler().fit_transform(vectors)
    embedding = reducer.fit_transform(scaled_vectors)
    embedding.shape

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(
        embedding[:, 0],
        embedding[:, 1])
    ax.gca().set_aspect('equal', 'datalim')
    ax.title('UMAP projection of the Vectors', fontsize=24)

    return fig
