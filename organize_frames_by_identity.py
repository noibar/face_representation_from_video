import argparse
import os
from os import path
from collections import Counter
from Track import MINIMUM_TRACKS_FOR_LABEL


def read_tracks_labels(labels_file):
    separator = " "
    tracks_labels = {}
    with open(labels_file) as f:
        lines = [line.strip().split(separator) for line in f]
        for line in lines:
            (tid, label) = line
            tracks_labels[tid] = label
    return tracks_labels


def get_frame_track(frame):
    return frame.split('_')[0]


def get_frames_by_track(frames):
    frames_by_track = {}
    for f in frames:
        t = get_frame_track(f)
        if t not in frames_by_track:
            frames_by_track[t] = []
        frames_by_track[t].append(f)
    return frames_by_track


def organize_frames_by_identity(frames_dir, output_dir, labels_file):
    track_labels = read_tracks_labels(labels_file)
    extracted_frames = os.listdir(frames_dir)
    frames_by_track = get_frames_by_track(extracted_frames)

    label_occurences = Counter(track_labels.values())
    for tid in track_labels:
        label = track_labels[tid]
        if label_occurences[label] < MINIMUM_TRACKS_FOR_LABEL:
            print(f'skip label {label}, number of tracks {label_occurences[label]}')
            continue

        if tid not in frames_by_track:
            print(f'skip track {tid}, no frames')
            continue

        label_dir = path.join(output_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        from shutil import copyfile
        for f in frames_by_track[tid]:
            src = path.join(frames_dir, f)
            dst = path.join(label_dir, f)
            copyfile(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', metavar='frames', type=str, nargs='?',
                        default='data/frames',
                        help='the pate of the frames directory')
    parser.add_argument('--output_dir', metavar='output', type=str, nargs='?',
                        default='data/frames',
                        help='the pate of the output directory')
    parser.add_argument('--labels_path', metavar='labels', type=str, nargs='?',
                        default='data/TheBigBangTheory.labels.txt',
                        help='the pate of the embedding file')
    args = parser.parse_args()

    organize_frames_by_identity(args.frames_dir, args.output_dir, args.labels_path)


if __name__ == "__main__":
    main()
