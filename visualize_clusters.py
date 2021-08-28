import xlsxwriter
import argparse
from Track import MINIMUM_TRACKS_FOR_LABEL


def read_tracks_times(tracks_file):
    separator = " "
    track_times = {}
    with open(tracks_file) as f:
        lines = [line.split(separator) for line in f]
        for line in lines:
            (T, identifier, left, top, right, bottom, status) = line
            if identifier not in track_times:
                track_times[identifier] = T
    return track_times


def read_labels_tracks(labels_file):
    separator = " "
    labels_tracks = {}
    with open(labels_file) as f:
        lines = [line.strip().split(separator) for line in f]
        for line in lines:
            (tid, label) = line
            if label not in labels_tracks:
                labels_tracks[label] = []
            labels_tracks[label].append(tid)
    return labels_tracks


def visualize_labels(episode_name, data_dir, tracks_file, labels_file):
    tracks_time = read_tracks_times(tracks_file)
    labels_tracks = read_labels_tracks(labels_file)
    # Create an new Excel file and add a worksheet.
    xlsx_name = f'{data_dir}/{episode_name}_clustered.xlsx'
    print('writing output to: ', xlsx_name)
    print('labels: ', labels_tracks.keys())
    workbook = xlsxwriter.Workbook(xlsx_name)
    for label in labels_tracks:
        worksheet = workbook.add_worksheet(label)
        worksheet.set_default_row(100)
        worksheet.write(f'A1', f'{label}')
        i = 2
        if len(labels_tracks[label]) < MINIMUM_TRACKS_FOR_LABEL:
            print(f'skipping visualization of {label}, only {labels_tracks[label]} tracks')
        for tid in labels_tracks[label]:
            worksheet.write(f'A{i}', f'{tid}')
            image = f'{data_dir}/frames/{tid}_{tracks_time[str(tid)]}_0.jpg'
            worksheet.insert_image(f'B{i}', image, {'x_scale': 0.5, 'y_scale': 0.5})
            i += 1

    workbook.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', metavar='episode', type=str, nargs='?',
                        default='TheBigBangTheory',
                        help='the name of the episode')
    parser.add_argument('--data_dir', metavar='dir', type=str, nargs='?',
                        default='data',
                        help='the data directory')
    parser.add_argument('--tracks_path', metavar='tracks', type=str, nargs='?',
                        default='data/TheBigBangTheory.track.txt',
                        help='the pate of the embedding file')
    parser.add_argument('--labels_path', metavar='labels', type=str, nargs='?',
                        default='data/TheBigBangTheory.labels.txt',
                        help='the pate of the embedding file')
    args = parser.parse_args()

    episode_name = args.episode
    visualize_labels(episode_name, args.data_dir, args.tracks_path, args.labels_path)


if __name__ == "__main__":
    main()
