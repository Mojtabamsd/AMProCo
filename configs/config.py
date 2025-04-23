import yaml
from pathlib import Path


class BaseConfig:
    def __init__(self, cpu=False, all_gpu=False, gpu_index=0):
        self.cpu = cpu
        self.all_gpu = all_gpu
        self.gpu_index = gpu_index


class SamplingConfig:
    def __init__(self, path_uvp5, path_uvp6, path_uvp6_csv, path_output, uvp_type, num_class, target_size,
                 sampling_method, sampling_percent_uvp5, sampling_percent_uvp6,
                 test_dataset_sampling, test_percent_uvp6, test_percent_uvp5, create_folder):
        self.path_uvp5 = path_uvp5
        self.path_uvp6 = path_uvp6
        self.path_uvp6_csv = path_uvp6_csv
        self.path_output = path_output
        self.uvp_type = uvp_type
        self.num_class = num_class
        self.target_size = target_size
        self.sampling_method = sampling_method
        self.sampling_percent_uvp5 = sampling_percent_uvp5
        self.sampling_percent_uvp6 = sampling_percent_uvp6
        self.test_dataset_sampling = test_dataset_sampling
        self.test_percent_uvp6 = test_percent_uvp6
        self.test_percent_uvp5 = test_percent_uvp5
        self.create_folder = create_folder


class TrainingContrastiveConfig:
    def __init__(self, dataset, im_factor, superclass_num, k_max, delta_min, architecture_type, batch_size,
                 accumulation_steps, num_workers, gray, target_size, padding, pre_train, learning_rate, weight_decay,
                 cos, momentum, schedule, num_epoch, warmup_epoch, twostage_epoch, loss, feat_dim,
                 temp, use_norm, path_pretrain):
        self.dataset = dataset
        self.im_factor = im_factor
        self.superclass_num = superclass_num
        self.k_max = k_max
        self.delta_min = delta_min
        self.architecture_type = architecture_type
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.num_workers = num_workers
        self.gray = gray
        self.target_size = target_size
        self.pre_train = pre_train
        self.padding = padding
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.cos = cos
        self.schedule = schedule
        self.num_epoch = num_epoch
        self.warmup_epoch = warmup_epoch
        self.twostage_epoch = twostage_epoch
        self.loss = loss
        self.feat_dim = feat_dim
        self.temp = temp
        self.use_norm = use_norm
        self.path_pretrain = path_pretrain


class PredictionConfig:
    def __init__(self, path_model, batch_size):
        self.path_model = path_model
        self.batch_size = batch_size


class Configuration:
    def __init__(self, config_file_path, input_path=None, output_path=None):
        with open(config_file_path, "r") as config_file:
            config_data = yaml.safe_load(config_file)

        self.input_path = input_path
        self.output_path = output_path
        self.base = BaseConfig(**config_data['base'])
        self.sampling = SamplingConfig(**config_data['sampling'])
        self.training_contrastive = TrainingContrastiveConfig(**config_data['training_contrastive'])
        self.prediction = PredictionConfig(**config_data['prediction'])

    def write(self, filename):
        filename = Path(filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True)
        with filename.open("w") as file_handler:
            yaml.dump(
                self, file_handler, allow_unicode=True, default_flow_style=False
            )

