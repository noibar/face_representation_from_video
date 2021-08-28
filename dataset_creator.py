import numpy as np
import random
import math
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from Track import CROPS

MIN_TRACKS_FOR_IDENTITY = 10
FALSE_ALARM_LABEL = "false_alarm"


def is_in_shots(shots, time):
    for shot in shots:
        if float(shot['start']) <= float(time) <= float(shot['end']):
            return True
    return False


def random_split_by_percentage(l, group_percentages):
    random.shuffle(l)
    l_parts = {}
    i = 0
    for group in group_percentages:
        p = group_percentages[group]
        n = min(math.ceil(len(l) * p), len(l))
        l_parts[group] = l[i:i + n]
        i = i + n
    return l_parts


class Dataset:
    def __init__(self, num_pairs, features_size, shuffle):
        self.pairs = np.empty((num_pairs, 2, features_size), dtype=float)
        self.pairs_labels = np.empty((num_pairs, 1), dtype=int)
        self.labels = np.empty((num_pairs, 2, 2), dtype=object)
        self.size = 0
        self.features_size = features_size
        self.shuffle = shuffle

    def add_pair(self, x1, x2, y1, y2, label):
        i = self.size
        self.pairs[i] = [x1, x2]
        self.pairs_labels[i] = label
        self.labels[i] = [y1, y2]
        self.size += 1

    def add_pairs(self, pairs):
        for p in pairs:
            x1, x2, y1, y2, label = p
            self.add_pair(x1, x2, y1, y2, label)

    def get_dataset(self):
        self.pairs.resize((self.size, 2, self.features_size))
        self.pairs_labels.resize((self.size, 1))
        self.labels.resize((self.size, 2, 2))

        dataset = list(zip(self.pairs, self.pairs_labels, self.labels))
        if self.shuffle:
            random.shuffle(dataset)
        pairs, pairs_labels, labels = zip(*dataset)

        return np.array(pairs), np.array(pairs_labels).astype('float32'), labels


class DatasetCreator:
    def __init__(self, tracks, shots, negative_by_distance=False, pos_pairs=2, neg_pairs=4, \
                 shuffle=True, num_crops=len(CROPS)):
        self.t = tracks
        self.shots = shots
        self.negative_by_distance = negative_by_distance
        self.pos_pairs = pos_pairs
        self.neg_pairs = neg_pairs
        self.wrong_tracks_negatives = []
        self.shuffle = shuffle
        self.num_crops = num_crops

    def create_datasets(self, dataset_parts={'train': 1}):
        datasets = {}
        datasets_shots = random_split_by_percentage(self.shots, dataset_parts)
        for dataset_name in datasets_shots:
            shots = datasets_shots[dataset_name]
            datasets[dataset_name] = self.create_dataset(shots)
        return datasets

    def filter_track_ids(self, track_ids):
        label_occurences = Counter(self.t.labels.values())
        track_ids = [track for track in track_ids if label_occurences[self.t.labels[track]] > MIN_TRACKS_FOR_IDENTITY
            and self.t.labels[track].lower() != FALSE_ALARM_LABEL]
        return track_ids

    def create_dataset(self, shots):
        track_ids = list(self.t.tracks.keys())
        track_ids = [tid for tid in track_ids if is_in_shots(shots, self.t.get_track_timestamps(tid)[0])]
        track_ids = self.filter_track_ids(track_ids)
        self.wrong_tracks_negatives = []

        num_frames = sum([len(self.t.tracks[tid]) for tid in track_ids])

        pairs = self.pos_pairs + self.neg_pairs
        num_pairs = pairs * num_frames * self.num_crops
        features_size = 2048
        dataset = Dataset(num_pairs, features_size, self.shuffle)
        dist = self.compute_track_distances(track_ids)

        n_for_track = {}
        frames_for_track = {}
        for tid in track_ids:
            track_frames = list(self.t.tracks[tid])
            frames_for_track[tid] = len(track_frames)
            n = 0
            for frame in track_frames:
                pairs = self.get_positive_pairs(tid, frame, track_frames.copy())
                n += len(pairs)
                dataset.add_pairs(pairs)

                pairs = self.get_negative_pairs(tid, frame, track_ids.copy(), dist)
                n += len(pairs)
                dataset.add_pairs(pairs)
            n_for_track[tid] = n

        label_occurences_in_dataset = dict(Counter([self.t.labels[tid] for tid in track_ids]))
        print('done creating dataset')
        print(f'\tSize: {dataset.size}')
        print(f'\tErrors: {len(self.wrong_tracks_negatives)}')
        print(f'\tCharacters: {label_occurences_in_dataset}')
        return dataset.get_dataset()

    def get_positive_pairs(self, tid, frame, track_frames):
        track_frames.remove(frame)
        random_couples = random.sample(track_frames, k=min(self.pos_pairs, len(track_frames)))
        x1_values = self.t.tracks[tid][frame][:self.num_crops]
        random.shuffle(x1_values)
        x2_values = [self.t.tracks[tid][frame][i] for frame in random_couples for i in range(self.num_crops) ]
        y1 = (tid, frame)
        y2_values = [(tid, frame) for i in range(self.num_crops) for frame in random_couples]
        pairs = [(x1, x2_values[i], y1, y2_values[i], 1) for i in range(len(random_couples)) for x1 in x1_values]
        return pairs

    def get_negative_pairs(self, tid, frame, track_ids, dist):
        simultanious_track_ids = [t for t in self.t.frames[frame]]
        simultanious_track_ids.remove(tid)
        random_couples = random.sample(simultanious_track_ids, k=min(self.neg_pairs, len(simultanious_track_ids)))
        x1_values = self.t.tracks[tid][frame][:self.num_crops]
        random.shuffle(x1_values)
        x2_values = [self.t.tracks[t][frame][i] for t in random_couples for i in range(self.num_crops)]
        y1 = (tid, frame)
        y2_values = [(t, frame) for t in random_couples for i in range(self.num_crops) ]
        if len(random_couples) == 0 and self.negative_by_distance:
            x2_dist, y2_dist = self.get_distant_track_frames(tid, track_ids, dist, self.neg_pairs - len(random_couples))
            x2_values.extend(x2_dist)
            y2_values.extend(y2_dist)
        return [(x1_values[i], x2_values[i], y1, y2_values[i], 0) for i in range(len(x1_values))]

    def get_distant_track_frames(self, tid, track_ids, dist, n, F=25):
        tracks_distances = dist[track_ids.index(tid)]
        F = min(F, max(int(len(tracks_distances) * 0.1), n))
        farthest_tracks_indexes = tracks_distances.argsort()[-F:][::-1]
        farthest_tracks = [track_ids[i] for i in farthest_tracks_indexes]
        track_ids = random.sample(farthest_tracks, n)

        t_label = self.t.labels[tid]
        x2 = []
        y2 = []
        for t in track_ids:
            frames = list(self.t.tracks[t].keys())
            frame = random.choice(frames)
            for i in range(min(len(self.t.tracks[t][frame]), self.num_crops)):
                x2.append(self.t.tracks[t][frame][i])
                y2.append((t, frame))

        for track in track_ids:
            if t_label == self.t.labels[track]:
                self.wrong_tracks_negatives.append(track)

        return x2, y2

    def compute_track_distances(self, track_ids):
        repr = self.t.get_tracks_representations(track_ids=track_ids, should_norm=False)
        track_base_representation = [repr[tid] for tid in track_ids]
        return squareform(pdist(track_base_representation, metric='euclidean'))
