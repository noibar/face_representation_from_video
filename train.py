from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import argparse
from siamse_model import create_siamese_mlp, create_mlp_embedding
import json
import random
from datetime import datetime
from os import path
import pickle
import tensorflow as tf
from Track import CROPS

from dataset_creator import DatasetCreator
from Track import Tracks

import matplotlib.image as mpimg


def visualize_datasets(images, labels, real_labels, title, output_dir, should_show=False):
    def show(ax, image, title):
        image = mpimg.imread(image)
        ax.set_title(f'{title}')
        ax.imshow(image)

    fig = plt.figure(figsize=(30, 9))
    fig.suptitle(title)

    n = 20
    axs = fig.subplots(4, n)
    for i in range(4):
        for j in range(n):
            ax = axs[i][j]
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_ticks([])

    imgs = 0
    for i in range(len(labels)):
        label = labels[i]
        if int(label) == 0:
            label = real_labels[i][0]
            show(axs[0, imgs], images[tuple(label)], label)
            label = real_labels[i][1]
            show(axs[1, imgs], images[tuple(label)],label)
            imgs += 1
        if imgs == n:
            break
    imgs = 0
    for i in range(len(labels)):
        label = labels[i]
        if int(label) == 1:
            label = real_labels[i][0]
            show(axs[2, imgs], images[tuple(label)], label)
            label = real_labels[i][1]
            show(axs[3, imgs], images[tuple(label)], label)
            imgs += 1
        if imgs == n:
            break

    axs[0, 0].set_ylabel('original')
    axs[1, 0].set_ylabel('different')
    axs[2, 0].set_ylabel('original')
    axs[3, 0].set_ylabel('same')
    plt.rcParams.update({'font.size': 8})
    plt.tight_layout()
    time = datetime.now().strftime("%m%d%Y_%H%M%S")
    print('save vis to: ', path.join(output_dir, 'visualize_dataset_{0}_{1}.png'.format(title, time)))
    plt.savefig(path.join(output_dir, 'visualize_dataset_{0}_{1}.png'.format(title, time)))
    if should_show:
        plt.show()
    else:
        plt.close()


def plt_metric(history, metric, title, output_dir):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        output_dir: the path of the output directory.

    Returns:
        None.
    """
    plt.plot(history[metric])
    plt.plot(history["val_" + metric])
    plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.savefig(path.join(output_dir, '{}.png'.format(title)))
    plt.show()


def train(datasets, checkpoint_path, epochs, batch_size, learning_rate, output_dir, t, network_type, repr_layer, weights):
    shape = (2048)
    train, val, test = datasets
    x_train_1, x_train_2, labels_train = train
    x_val_1, x_val_2, labels_val = val
    x_test_1, x_test_2, labels_test = test

    siamese = create_siamese_mlp(shape, network_type, learning_rate, weights_path=weights)
    if not weights:
        weights_path = f'{output_dir}/weights_{epochs}_{batch_size}_{network_type}.data'
        weights = siamese.get_weights()
        with open(weights_path, 'wb') as file:
            print('writing weights to: ', file)
            pickle.dump(weights, file)
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          save_weights_only=True,
                                          verbose=1)

    history = siamese.fit(
        [x_train_1, x_train_2],
        labels_train,
        shuffle=True,
        validation_data=([x_val_1, x_val_2], labels_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint_callback]
    )

    time = datetime.now().strftime("%m%d%Y_%H%M%S")
    # Plot the accuracy
    plt_metric(history=history.history, metric="accuracy", title=f"Model accuracy {batch_size} {time}", output_dir=output_dir)

    # Plot the constrastive loss
    plt_metric(history=history.history, metric="loss", title=f"Constrastive Loss {batch_size} {time}", output_dir=output_dir)

    def create_feature_func(embedding):
        def feature_func(features):
            features = tf.convert_to_tensor(features)
            e_f = embedding.predict(features)
            return e_f
        return feature_func

    print(siamese.evaluate([x_test_1, x_test_2], labels_test))

    # Save tracks new embedding
    embedding, repr_layer = create_mlp_embedding(siamese, repr_layer)
    feature_func = create_feature_func(embedding)
    embeded_track_features = t.get_tracks_representations(features_func=feature_func, should_norm=True)
    if learning_rate:
        embedding_features_path = f'{output_dir}/tracks_{epochs}_{batch_size}_{network_type}_{learning_rate}' \
                                  f'_{repr_layer}.embeded.data'
    else:
        embedding_features_path = f'{output_dir}/tracks_{epochs}_{batch_size}_{network_type}_{repr_layer}.embeded.data'
    with open(embedding_features_path,'wb') as ef:
        print('save tracks new embedding to: ', embedding_features_path)
        pickle.dump(embeded_track_features, ef)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks', metavar='tracks', type=str, nargs='?',
                        help='the path of the tracks file')
    parser.add_argument('--shots', metavar='shots', type=str, nargs='?',
                        help='the path of the tracks file')
    parser.add_argument('--epochs', metavar='epochs', type=int, nargs='?',
                        default=10,
                        help='number of epochs to learn from')
    parser.add_argument('--negative', metavar='negative_pairs', type=int, nargs='?',
                        default=4,
                        help='number of negative pairs to use. on small dataset consider using smaller number.')
    parser.add_argument('--batch', metavar='batch', type=int, nargs='?',
                        help='size of batches to learn from')
    parser.add_argument('--learning_rate', metavar='learning_rate', type=float, nargs='?',
                        help='learning rate of sgd, if none learning with other optimizer')
    parser.add_argument('--network_type', metavar='network', type=str, nargs='?',
                        default='small')
    parser.add_argument('--checkpoint_path', metavar='checkpoint_path', type=str, nargs='?',
                        default='checkpoint/training_{}_{}_{}_{}/cp.ckpt',
                        help='path for saving checkpoints')
    parser.add_argument('--output', metavar='output', type=str, nargs='?',
                        default='output',
                        help='path for saving output')
    parser.add_argument('--weights', metavar='weights', type=str, nargs='?',
                        help='path for initial weights file to load')
    parser.add_argument('--negative_by_distance', action='store_true', default=False)
    parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle frames before learning')
    parser.add_argument('--repr_layer', metavar='repr_layer', type=int, nargs='?',
                        help='index of representation layer')
    parser.add_argument('--num_crops', metavar='crops', type=int, nargs='?',
                        default=len(CROPS),
                        help=f'number of crops to use as a form of data augmentation, max is {len(CROPS)}')

    random.seed(datetime.now())

    args = parser.parse_args()
    tracks_file = args.tracks
    epochs = args.epochs
    batch_size = args.batch
    t = Tracks()
    t.load(tracks_file)
    shots_file = open(args.shots, 'r')
    shots = json.load(shots_file)

    creator = DatasetCreator(t, shots['content'], negative_by_distance=args.negative_by_distance,
                             neg_pairs=args.negative, shuffle=args.shuffle, num_crops=args.num_crops)
    datasets = creator.create_datasets({'train': 0.7, 'validate': 0.15, 'test': 0.15})
    print('done creating datasets')

    x_train, labels_train, real_labels_train = datasets['train']
    x_val, labels_val, real_labels_val = datasets['validate']
    x_test, labels_test, real_labels_test = datasets['test']

    # Change the data type to a floating point format
    output_dir = args.output
    x_train_1, x_train_2 = x_train[:, 0], x_train[:, 1]
    x_val_1, x_val_2 = x_val[:, 0], x_val[:, 1]
    x_test_1, x_test_2 = x_test[:, 0], x_test[:, 1]
    visualize_datasets(t.img, labels_train, real_labels_train, 'train', output_dir, should_show=True)
    visualize_datasets(t.img, labels_val, real_labels_val, 'validation', output_dir)
    visualize_datasets(t.img, labels_test, real_labels_test, 'test', output_dir)
    print(f'training with {len(labels_train)} pairs, validating with {len(labels_val)}')
    datasets = [(x_train_1, x_train_2, labels_train), (x_val_1, x_val_2, labels_val), (x_test_1, x_test_2, labels_test)]

    learning_description = args.learning_rate if args.learning_rate is not None else 'adam'
    print('learning: ', learning_description)
    if batch_size is not None:
        checkpoint_path = args.checkpoint_path.format(epochs, batch_size, learning_description, args.network_type)
        train(datasets, checkpoint_path, epochs, batch_size, args.learning_rate, args.output, t, args.network_type, args.repr_layer, args.weights)
        return
    for batch_size in [500, 1000, 2000]:
        checkpoint_path = args.checkpoint_path.format(epochs, batch_size, learning_description, args.network_type)
        train(datasets, checkpoint_path, epochs, batch_size, args.learning_rate, args.output, t, args.network_type, args.repr_layer, args.weights)


if __name__ == '__main__':
    main()
