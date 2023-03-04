
class ShanghaiA:
    def __init__(self):
        self.train_set_dir = '/Users/danielpirak/Projects/crowd-density/CSRNet/datasets/shanghaitech_A/train_data'
        self.test_set_dir = '/Users/danielpirak/Projects/crowd-density/CSRNet/datasets/shanghaitech_A/train_data'
        self.eval_interval = 3
        self.n_epochs_no_improvement = 3
        self.max_train_epochs = 1000
        self.learning_rate = 1e-3
