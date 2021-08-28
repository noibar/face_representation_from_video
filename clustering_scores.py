import bcubed
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.metrics as metrics

from sklearn.decomposition import PCA


def calculate_cluster_accuracy(labels, cluster_labels):
    N = len(labels)
    clusters = list(set(cluster_labels))
    C = len(clusters)
    cluster_labels = list(cluster_labels)

    acc_score = 0
    cluster_labels_count = {}
    for i in range(len(cluster_labels)):
        if cluster_labels[i] not in cluster_labels_count:
            cluster_labels_count[cluster_labels[i]] = {}
        if labels[i] not in cluster_labels_count[cluster_labels[i]]:
            cluster_labels_count[cluster_labels[i]][labels[i]] = 0
        cluster_labels_count[cluster_labels[i]][labels[i]] += 1

    p_C = {}
    for i in range(C):
        most_popular_count = 0
        for l in cluster_labels_count[i]:
            if cluster_labels_count[i][l] > most_popular_count:
                most_popular_count = cluster_labels_count[i][l]
        p_C[i] = most_popular_count / cluster_labels.count(clusters[i])

    for i in range(C):
        n_c = cluster_labels.count(clusters[i])
        p_c = p_C[i]  # largest number of samples from the same label to nc
        acc_score += p_c * n_c
    acc_score /= N

    return acc_score


def get_clusters_scores(labels, clusters, track_ids, noFalseCluster=False):
    cdict = {}
    ldict = {}
    for i in range(len(clusters)):
        ldict[track_ids[i]] = set([labels[i]])
        cdict[track_ids[i]] = set([clusters[i]])

    scores = {}
    precision = bcubed.precision(cdict, ldict)
    scores['precision'] = precision
    recall = bcubed.recall(cdict, ldict)
    scores['recall'] = recall
    scores['fscore'] = bcubed.fscore(precision, recall)

    scores['homogeneity'] = metrics.homogeneity_score(labels, clusters)
    scores['completeness'] = metrics.completeness_score(labels, clusters)
    scores['rand_score'] = metrics.rand_score(labels, clusters)

    if noFalseCluster:
        accuracy = calculate_cluster_accuracy(labels, clusters)
        scores['accuracy'] = accuracy

    return scores


def scatter_thumbnails(data, images, zoom=0.12, colors=None):
    assert len(data) == len(images)

    # reduce embedding dimentions to 2
    x = PCA(n_components=2).fit_transform(data) if len(data[0]) > 2 else data

    # create a scatter plot.
    f = plt.figure(figsize=(22, 15))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], s=4)
    _ = ax.axis('off')
    _ = ax.axis('tight')

    # add thumbnails :)
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    for i in range(len(images)):
        image = plt.imread(images[i])
        im = OffsetImage(image, zoom=zoom)
        bboxprops = dict(edgecolor=colors[i]) if colors is not None else None
        ab = AnnotationBbox(im, x[i], xycoords='data',
                            frameon=(bboxprops is not None),
                            pad=0.02,
                            bboxprops=bboxprops)
        ax.add_artist(ab)
    return ax


def plot_clusters(data, images, algorithm, *args, **kwds):
    labels = algorithm(*args, **kwds).fit_predict(data)
    palette = sns.color_palette('deep', np.max(labels) + 1)
    colors = [palette[x] if x >= 0 else (0,0,0) for x in labels]
    ax = scatter_thumbnails(data, images, 0.2, colors)
    plt.title(f'Clusters found by {algorithm.__name__}')
    return labels


def get_labels_dict(clusters, labels):
    labels_dict= {}
    for i in range(len(clusters)):
        if labels[i] not in labels_dict:
            labels_dict[labels[i]] = []
        labels_dict[labels[i]].append(clusters[i])
    return labels_dict
