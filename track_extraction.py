import cv2
import pickle
from pyannote.video import Video
import argparse
from os import path
from PIL import Image
from Track import Tracks, CROPS

from img2vec_pytorch import Img2Vec

target_shape = (225, 225)


def get_tracks_labels(labels_file):
    separator = ' '
    labels = {}
    if labels_file is not "":
        with open(labels_file) as f:
            for lines in f:
                line = lines.strip().split(separator)
                labels[line[0]] = line[1]
    print('track unique labels: ', set(list(labels.values())))
    return labels


def get_image(filename):
    return Image.open(filename)


def get_face_crop(frame, bb, frame_height, frame_width, crop):
    (left, top, right, bottom) = bb
    (sub_left, sub_top, add_right, add_bottom) = crop
    left, top, right, bottom = float(left), float(top), float(right), float(bottom)
    width = right - left
    height = bottom - top

    left = int((left - (sub_left*width)) * frame_width)
    right = int((right + (add_right*width)) * frame_width)
    top = int((top - (sub_top*height)) * frame_height)
    bottom = int((bottom + (add_bottom * height)) * frame_height)
    image = frame[max(top,0):bottom,max(left,0):right,:]

    return cv2.resize(image, dsize=target_shape, interpolation=cv2.INTER_CUBIC)


def get_tracks_to_skip(labels, track_ids):
    if len(labels) == 0:
        return []

    skip = []
    missing = []
    for tid in track_ids:
        if tid not in labels:
            missing.append(tid)
        if tid not in labels or labels[tid].lower() == "false_alarm":
            skip.append(tid)
    print(f'following track ids: {missing} are not labeled in the labels file')
    return skip


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


def to_float(area):
    a, b, c, d = area
    return float(a), float(b), float(c), float(d)


def area(a):
    dx = a[2] - a[0]
    dy = a[3] - a[1]
    return dx*dy


def intersection_area(a, b):  # returns None if rectangles don't intersect
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx >= 0) and (dy >= 0):
        return dx*dy


def filter_overlapping_faces(faces, max_overlap=0.5):
    filtered_faces = []
    for i in range(len(faces)):
        if i in filtered_faces:
            continue
        for j in range(i + 1, len(faces)):
            if j in filtered_faces:
                continue
            _, first = faces[i]
            _, second = faces[j]
            first, second = to_float(first), to_float(second)
            intersection = intersection_area(first, second)
            if intersection is not None:
                if intersection > max_overlap * min(area(first), area(second)):
                    if area(first) > area(second):
                        filtered_faces.append(j)
                    else:
                        filtered_faces.append(i)
                        break

    filtered_faces_tracks = [faces[i][0] for i in filtered_faces]
    if len(filtered_faces_tracks) > 0:
        print('skip overlapping tracks:', filtered_faces_tracks)
    faces = [faces[i] for i in range(len(faces)) if i not in filtered_faces]
    return faces


def extract_tracks(video_file, tracks_file, labels, tracks_output_file, data_dir, visualize, model, cuda, feature_size, frames_only, tracks_frames_file):
    video = Video(video_file)
    frame_width, frame_height = video.frame_size

    face_by_time = read_tracks(tracks_file)
    last_frame_faces = []
    track_frames = {}
    for timestamp, rgb in video:
        timestamp = "{:.3f}".format(timestamp)
        if timestamp not in face_by_time:
            continue
        faces = face_by_time[timestamp]
        for face in faces:
            tid, area = face
            if tid not in last_frame_faces:
                print(f'track {tid}: {timestamp} start extracting frames')

        faces = filter_overlapping_faces(faces)
        last_frame_faces = []
        for face in faces:
            tid, area = face
            last_frame_faces.append(tid)
            try:
                for crop in CROPS:
                    image = get_face_crop(rgb, area, frame_height, frame_width, crop)
                    image = Image.fromarray(image, 'RGB')
                    img_name = '{0}_{1}_{2}.jpg'.format(tid, timestamp, CROPS.index(crop))
                    img_path = path.join(data_dir, "frames", img_name)
                    image.save(img_path)
                    if tid not in track_frames:
                        track_frames[tid] = []
                    track_frames[tid].append((timestamp, img_path))
                    if visualize:
                        cv2.imshow('image_{}'.format(tid), image)
                        cv2.waitKey(1)
            except Exception as e:
                print('failed writing face_crop image', e, '(track {0}, time {1})'.format(tid, timestamp))

    with open(tracks_frames_file, 'wb') as frames_file:
        pickle.dump(track_frames, frames_file)
    if not frames_only:
        save_track_features(track_frames, labels, tracks_output_file, model, cuda, feature_size)


def save_track_features(track_frames, labels, tracks_output_file, model, cuda, features_size):
    t = Tracks(features_size)
    features_extractor = Img2Vec(cuda=cuda, model=model, layer_output_size=features_size)

    track_ids = list(track_frames.keys())
    skipping_track_ids = get_tracks_to_skip(labels, track_ids)
    print('skipping unlabeled or false_alarm tracks: ', skipping_track_ids)
    from collections import Counter
    label_occurences = Counter(labels.values())

    for tid in track_frames:
        if tid in skipping_track_ids:
            continue
        if label_occurences[labels[tid]] < 5:
            print(f'skipping track {tid} with label {labels[tid]}, only {label_occurences[labels[tid]]} tracks')
            continue
        track_images = []
        timestamp, img_path = track_frames[tid][0]
        print(f'track {tid}: {timestamp} start extracting features')
        for track_frame in track_frames[tid]:
            timestamp, img_path = track_frame
            track_images.append(get_image(img_path))
        track_features = features_extractor.get_vec(track_images)
        for i in range(len(track_features)):
            f = track_features[i]
            timestamp, img_path = track_frames[tid][i]
            label = ""
            if tid in labels:
                label = labels[tid]
            t.add_track_features(tid, timestamp, f, label, img_path)

    t.save(tracks_output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', metavar='video_path', type=str, nargs='?',
                        help='the path of the video')
    parser.add_argument('--tracks', metavar='tracks_path', type=str, nargs='?',
                        help='the path of the tracks')
    parser.add_argument('--data_dir', metavar='data', type=str, nargs='?',
                        help='the data directory')
    parser.add_argument('--tracks_file', metavar='tracks_file', type=str, nargs='?',
                        help='the path for saving the created Tracks object')
    parser.add_argument('--labels', metavar='labels', type=str, nargs='?',
                        default='',
                        help='the path of the labels file')
    parser.add_argument('--frames_only', action='store_true', default=False,
                        help='extract frames images only, without extracting features of each frame')
    parser.add_argument('--features_only', action='store_true', default=False,
                        help='extract features only, take tracks frames object from track_frames_path')
    parser.add_argument('--track_frames_path', metavar='track_frames_path', type=str, nargs='?',
                        help='the path of the frames data file')
    parser.add_argument('--model', metavar='model', type=str, nargs='?', default='resnet50',
                        help='the model we want to use for feature extraction')

    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--feature_size', type=int, default=2048)

    args = parser.parse_args()

    video_file = args.video
    tracks_file = args.tracks
    labels_file = args.labels
    tracks_output_file = args.tracks_file

    print('track extraction input:')
    print('\tvideo:', video_file)
    print('\ttracks:', tracks_file)
    print('\tlabels:', labels_file)
    print('track extraction writing output to:\n\t', tracks_output_file)
    labels = get_tracks_labels(labels_file)

    if args.features_only:
        with open(args.track_frames_path, 'rb') as track_frames_file:
            track_frames = pickle.load(track_frames_file)
            save_track_features(track_frames, labels, tracks_output_file, args.model, args.cuda, args.feature_size)
    else:
        extract_tracks(video_file, tracks_file, labels, tracks_output_file, args.data_dir, args.visualize, args.model, args.cuda, args.feature_size,
                       args.frames_only, args.track_frames_path)


if __name__ == '__main__':
    main()
