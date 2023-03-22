
class ShanghaiA:
    def __init__(self):
        self.train_set_dir = '/Users/danielpirak/Projects/ML_goofing/crowd-density/CSRNet/datasets/shanghaitech_A/train_data'
        self.test_set_dir = '/Users/danielpirak/Projects/ML_goofing/crowd-density/CSRNet/datasets/shanghaitech_A/test_data'
        self.n_random_crops = 5
        self.eval_interval = 3
        self.n_epochs_no_improvement = 3
        self.max_train_epochs = 1000
        self.learning_rate = 1e-4
