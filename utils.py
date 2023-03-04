from os import path
from datetime import datetime
import sys
import json


class Names:
    def __init__(self):
        self.images = 'images'
        self.ground_truths = 'ground_truth'
        self.densities = 'densities'
        self.train_data = 'train_data'
        self.test_data = 'test_data'
        self.shanghaitech_A = 'shanghaitech_A'
        self.shanghaitech_B = 'shanghaitech_B'
        self.datasets = 'datasets'
        self.output_folder = 'output'
        self.log_file = 'log'
        self.model_file = 'model.pth'
        self.args_files = 'args.json'


names = Names()


def get_dataset_dirs(dataset_name):
    dataset_dir = path.join(path.dirname(__file__), names.datasets, dataset_name)
    images_train_dir = path.join(dataset_dir, names.train_data, names.images)
    ground_truth_train_dir = path.join(dataset_dir, names.train_data, names.ground_truths)
    densities_train_dir = path.join(dataset_dir, names.train_data, names.densities)

    images_test_dir = path.join(dataset_dir, names.test_data, names.images)
    ground_truth_test_dir = path.join(dataset_dir, names.test_data, names.ground_truths)
    densities_test_dir = path.join(dataset_dir, names.test_data, names.densities)
    dirs_dict = {names.train_data:[images_train_dir, ground_truth_train_dir, densities_train_dir],
                 names.test_data:[images_test_dir, ground_truth_test_dir, densities_test_dir]}
    return dirs_dict


class Logger(object):
    def __init__(self, path, out_type):
        self.terminal = out_type
        self.log = open(path, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def redirect_output(path):
    sys.stdout = Logger(path, sys.stdout)
    sys.stderr = Logger(path, sys.stderr)


def get_root_path():
    return path.dirname(path.realpath(__file__))


def get_time():
    return datetime.today().strftime('%d_%m_%Y__%H_%M_%S_%f')[:-3]


def get_shanghai_image_name(sample):
    return 'IMG_{}.jpg'.format(sample)


def get_shanghai_gt_name(sample_idx):
    return 'GT_IMG_{}.mat'.format(sample_idx)


def get_density_name(sample_idx):
    return 'IMG_{}.npy'.format(sample_idx)


def log_args(output_dir, args):
    with open(path.join(output_dir, names.args_files), 'w') as f:
        json.dump(args.__dict__, f, indent=4, separators=(',', ': '))