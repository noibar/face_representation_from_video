import os
import seaborn as sns
from tensorflow.keras import metrics
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from siamse_model import create_siamese_mlp, create_mlp_embedding
from tensorflow.train import latest_checkpoint
import tensorflow as tf
from Track import Tracks
import pickle


def calculate_tracks_similarity(tracks_features, tracks_labels):
    positive_cosine = []
    negative_cosine = []
    cosine_similarity = metrics.CosineSimilarity()
    for i in range(len(tracks_features)):
        for j in range(i + 1, len(tracks_features)):
            cos_sim = cosine_similarity(tracks_features[i], tracks_features[j])
            if tracks_labels[i] == tracks_labels[j]:
                positive_cosine.append(cos_sim.numpy())
            else:
                negative_cosine.append(cos_sim.numpy())
    return positive_cosine, negative_cosine


def draw_histogram(pos, neg, title, output_dir):
    sns.set(style="darkgrid")

    sns.histplot(data=pos, color="skyblue", label="Positive", kde=True)
    sns.histplot(data=neg, color="red", label="Negative", kde=True)

    plt.legend()
    plt.savefig(os.path.join(output_dir, '{}_histogram.png'.format(title)))
    plt.show()


def plot_dendogram(features, labels, title, subtitle, output):
    plt.figure(figsize=(20, 15))
    plt.title(title)
    linkage = shc.linkage(features, method='ward')
    dend = shc.dendrogram(linkage, labels=labels, leaf_rotation=0, orientation="left", color_threshold=3,
                          above_threshold_color='grey')
    ax = plt.gca()
    xlbls = ax.get_ymajorticklabels()
    labels_unique = list(set(labels))
    my_palette = plt.cm.get_cmap("Accent", len(labels_unique))
    for lbl in xlbls:
        value = lbl.get_text()
        lbl.set_color(my_palette(labels_unique.index(value)))

    plt.rcParams.update({'font.size': 10})
    plt.tight_layout()
    plt.savefig(os.path.join(output, f'{title} {subtitle}.png'))
    plt.show()

def create_feature_func(embedding):
    def feature_func(features):
        features = tf.convert_to_tensor(features)
        e_f = embedding.predict(features)
        return e_f

    return feature_func


def create_tracks_embedding(tracks_file, network_type, checkpoint_dir, repr_layer, embedding_features_path):
    t = Tracks()
    t.load(tracks_file)
    shape = (2048)

    print('loading siamese trained network')
    siamese_model = create_siamese_mlp(shape, network_type)
    latest = latest_checkpoint(checkpoint_dir)
    siamese_model.load_weights(latest)

    # inspect what the network has learned.
    print('getting tracks embedded representation')
    embedding, _ = create_mlp_embedding(siamese_model, repr_layer)
    feature_func = create_feature_func(embedding)
    embeded_track_features = t.get_tracks_representations(features_func=feature_func, should_norm=True)
    with open(embedding_features_path,'wb') as ef:
        pickle.dump(embeded_track_features, ef)
