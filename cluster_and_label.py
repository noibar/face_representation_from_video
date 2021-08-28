from pyannote.video.face.clustering import FaceClustering
from pyannote.video import Video
from PIL import Image
import numpy as np
import cv2 as cv
import argparse
import pickle
from datetime import datetime
import math
import matplotlib.pyplot as plt
from os import path
import random
from collections import Counter
from Track import MINIMUM_TRACKS_FOR_LABEL

def show_multiple_images(images):
    def show(ax, image):
        ax.imshow(image)

    fig = plt.figure(figsize=(15, 5))

    n = min(len(images), 10)
    images = random.sample(images, n)
    axs = fig.subplots(1,n)
    for i in range(n):
        ax = axs[i]
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_ticks([])
        show(ax, images[i])

    plt.rcParams.update({'font.size': 8})
    plt.tight_layout()
    plt.show()


def split_embedding_file(embedding_file, max_num_of_tracks):
    '''
    Split the given embedding file to files that contains given max number of tracks.
    '''
    if max_num_of_tracks is None:
        return [embedding_file]

    files = []
    with open(embedding_file, 'r') as file:
        lines = [line for line in file]
        line_tracks = [line.split()[1] for line in lines]
        tracks = set(line_tracks)

        num_of_parts = math.ceil(len(tracks)/float(max_num_of_tracks))
        parts = {}
        for i in range(1, num_of_parts+1):
            parts[i] = []
        for i in range(len(lines)):
            num_part = int(int(line_tracks[i])/max_num_of_tracks)
            parts[num_part + 1].append(lines[i])

        for num_part in parts:
            part_file_path = '{0}.{1}'.format(embedding_file, str(num_part))
            lines = parts[num_part]
            with open(part_file_path, 'w') as part_file:
                for line in lines:
                    part_file.write(line)
                files.append(part_file_path)

    return files


def read_tracks(tracks_file):
    separator = " "
    faces_by_time = {}
    with open(tracks_file) as f:
        lines = [line.split(separator) for line in f]
        for line in lines:
            (T, identifier, left, top, right, bottom, status) = line
            if T not in faces_by_time:
                faces_by_time[T] = []
            faces_by_time[T].append((identifier, (left, top, right, bottom)))
    return faces_by_time


def get_face_crop(frame, bb, frame_height, frame_width):
    target_shape = (225,225)
    (left, top, right, bottom) = bb
    left = int(float(left) * frame_width)
    right = int(float(right) * frame_width)
    top = int(float(top) * frame_height)
    bottom = int(float(bottom) * frame_height)

    area = (left, top, right, bottom)
    image = Image.fromarray(frame, mode='RGB')
    image = image.crop(area)
    image = np.array(image)
    return cv.resize(image, dsize=target_shape, interpolation=cv.INTER_CUBIC)


def extract_tracks(video_file, tracks_file, labels, track_labels, frames_dir, num_of_extracted_frames=20):
    '''
    :return: Returns images of the given tracks ids, and extract first frame of every track for validation.
    '''
    video = Video(video_file)
    frame_width, frame_height = video.frame_size

    face_by_time = read_tracks(tracks_file)
    face_by_tracks = {}
    saved_tracks_first_frame = {}
    for timestamp, rgb in video:
        timestamp = "{:.3f}".format(timestamp)
        if timestamp not in face_by_time:
            continue
        faces = face_by_time[timestamp]
        for face in faces:
            tid, area = face
            tid = int(tid)
            try:
                if track_labels[tid] not in face_by_tracks:
                    face_by_tracks[track_labels[tid]] = []
                if tid in labels:
                    if len(face_by_tracks[tid]) > num_of_extracted_frames:
                        continue
                else:
                    if tid in saved_tracks_first_frame:
                        continue

                image = get_face_crop(rgb, area, frame_height, frame_width)
                if tid not in saved_tracks_first_frame:
                    img_name = f'{tid}_{timestamp}.jpg'
                    img_path = path.join(frames_dir, img_name)
                    cv.imwrite(img_path, image)
                    saved_tracks_first_frame[tid] = True

                face_by_tracks[track_labels[tid]].append(image)
            except Exception as e:
                # TODO: part of these happen because tracks embedding is partial or not found, and then there are no present
                # in the results.
                print('failed cropping face_crop image', e, '(track {0}, time {1})'.format(tid, timestamp))

    return face_by_tracks


def rename_labels(results, episode_name, suffix, data_dir, tracks_file, labels_file, frames_dir):
    video_file = f'{data_dir}/{episode_name}.{suffix}'
    tracks_file = f'{tracks_file}'
    print(f'read video file from {video_file}, tracks file from {tracks_file}')

    labels = []
    track_labels = {}
    for result in results:
        print('add: ', result.labels())
        labels.extend(result.labels())
        for _, track_id, cluster in result.itertracks(yield_label=True):
            track_labels[track_id] = cluster
    print(f'need to tag {len(labels)} labels.')

    faces_by_tracks = extract_tracks(video_file, tracks_file, labels, track_labels, frames_dir)

    label_occurences = Counter(track_labels.values())
    mapping = {}
    for label in labels:
        if label_occurences[label] < MINIMUM_TRACKS_FOR_LABEL:
            mapping[label] = "false_alarm"
            print(f'skipping label: {label} with {label_occurences[label]} tracks')
            continue

        if label not in faces_by_tracks:
            print('missing label track: ', label)
            continue

        show_multiple_images(faces_by_tracks[label])
        name = input('Enter identity name: ')
        mapping[label] = name

    results = [result.rename_labels(mapping=mapping) for result in results]

    print(f'writing labels to {labels_file}')
    with open(labels_file, 'w') as fp:
        for result in results:
            for _, track_id, cluster in result.itertracks(yield_label=True):
                fp.write(f'{track_id} {cluster}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clusters_path', metavar='clusters_path', type=str, nargs='?',
                        default='clustering_results.pickle.{}'.format(datetime.now().strftime("%m%d%Y%H%M")),
                        help='the path of the clustering results')
    parser.add_argument('--load_cached', action='store_true', default=False,
                        help='load existing clustering results, useful for relabeling')
    parser.add_argument('--episode', metavar='episode', type=str, nargs='?',
                        default='TheBigBangTheory',
                        help='the name of the episode')
    parser.add_argument('--suffix', metavar='suffix', type=str, nargs='?',
                        default='mkv',
                        help='the name of the episode')
    parser.add_argument('--data_dir', metavar='data_dir', type=str, nargs='?',
                        default='data',
                        help='the path of the data directory')
    parser.add_argument('--frames_dir', metavar='frames', type=str, nargs='?',
                        default='frames',
                        help='the path of the frames directory (that will be inside the data dir)')
    parser.add_argument('--embedding_path', metavar='embedding_path', type=str, nargs='?',
                        help='the pate of the embedding file')
    parser.add_argument('--tracks_path', metavar='tracks', type=str, nargs='?',
                        help='the pate of the tracks file')
    parser.add_argument('--labels_path', metavar='labels', type=str, nargs='?',
                        help='the pate of the labels file')
    parser.add_argument('--max', metavar='max_tracks', type=int, nargs='?',
                        help='max nums of tracks in embedding file')
    args = parser.parse_args()

    episode_name = args.episode
    if not args.load_cached:
        embedding_file = args.embedding_path
        files = split_embedding_file(embedding_file, max_num_of_tracks=args.max)
        results = []
        for file in files:
            clustering = FaceClustering(threshold=0.6)
            face_tracks, embeddings = clustering.model.preprocess(file)
            result = clustering(face_tracks, features=embeddings)
            results.append(result)
        with open(args.clusters_path, 'wb') as f:
            print('save clustering result to: ', args.clusters_path)
            pickle.dump(results, f)

    else:
        with open(args.clusters_path, 'rb') as f:
            results = pickle.load(f)

    rename_labels(results, episode_name, args.suffix, args.data_dir, args.tracks_path, args.labels_path,
                  path.join(args.data_dir, args.frames_dir))


if __name__ == "__main__":
    main()
