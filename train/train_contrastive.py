import datetime
from configs.config import Configuration
from tools.console import Console
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from dataset.uvp_dataset import UvpDataset
from models import resnext
from models import resnet_cifar
from dataset.cifar import IMBALANCECIFAR100
from tools.autoaug import CIFAR10Policy, Cutout
import math
import os
import shutil
import torch
import torch.distributed as dist
from tools.utils import report_to_df, plot_loss, shot_acc
from tools.randaugment import rand_augment_transform
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomAffine, RandomResizedCrop, \
    ColorJitter, RandomGrayscale, RandomPerspective, RandomVerticalFlip
from tools.augmentation import GaussianNoise, ResizeAndPad
from models.loss import LogitAdjust
from models.proco import ProCoLoss
from models.amproco import HierarchicalProCoWrapper
from dataset.cifar import CIFAR100_SUPERCLASSES
import time
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from tools.visualization import plot_tsne_from_validate


def train_contrastive(config_path, input_path, output_path):

    config = Configuration(config_path, input_path, output_path)
    config.phase = 'train'      # will train with whole dataset and testing results if there is a test file
    # phase = 'train_val'  # will train with 80% dataset and testing results with the rest 20% of data

    config.input_path = input_path

    # Create output directory
    input_folder = Path(input_path)
    output_folder = Path(output_path)

    input_folder_train = input_folder
    input_folder_test = input_folder

    console = Console(output_folder)
    console.info("Training started ...")

    sampled_images_csv_filename1 = "sampled_images_train.csv"
    sampled_images_csv_filename2 = "sampled_images_test.csv"
    input_csv_train = input_folder_train / sampled_images_csv_filename1
    input_csv_test = input_folder_test / sampled_images_csv_filename2
    input_csv_val = input_folder_test / sampled_images_csv_filename2

    config.input_folder_train = str(input_folder_train)
    config.input_folder_test = str(input_folder_test)
    config.input_csv_train = str(input_csv_train)
    config.input_csv_test = str(input_csv_test)
    config.input_csv_val = str(input_csv_val)

    if config.training_contrastive.dataset == 'uvp':
        if not input_csv_train.is_file():
            console.info("Label not provided for training")
            print(input_csv_train)

        if not input_csv_test.is_file():
            console.info("Label not provided for testing")
            print(input_csv_test)

    if config.training_contrastive.path_pretrain:
        training_path = Path(config.training_contrastive.path_pretrain)
        config.training_path = training_path
    else:
        time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        rel_training_path = Path(config.training_contrastive.dataset + "_training_contrastive" + time_str)
        training_path = output_folder / rel_training_path
        config.training_path = training_path
        if not training_path.exists():
            training_path.mkdir(exist_ok=True, parents=True)
        elif training_path.exists():
            console.error("The output folder", training_path, "exists.")
            console.quit("Folder exists, not overwriting previous results.")

    # Save configuration file
    output_config_filename = training_path / "config.yaml"
    config.write(output_config_filename)

    config.training_path = str(training_path)

    # parallel processing
    # config.world_size = torch.cuda.device_count()

    if config.base.all_gpu:
        world_size = torch.cuda.device_count()
        console.info(f"Number of GPU available:  {world_size}")
    else:
        world_size = 1

    # dist.init_process_group(backend='gloo', init_method='env://', world_size=config.world_size, rank=rank)

    if config.training_contrastive.dataset == 'uvp':
        if world_size > 1:
            mp.spawn(train_uvp, args=(world_size, config, console), nprocs=world_size, join=True)
        else:
            train_uvp(config.base.gpu_index, world_size, config, console)

    elif config.training_contrastive.dataset == 'cifar100':
        if world_size > 1:
            mp.spawn(train_cifar, args=(world_size, config, console), nprocs=world_size, join=True)
        else:
            train_cifar(config.base.gpu_index, world_size, config, console)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if world_size > 1:
        dist.init_process_group(
            backend='nccl',  # Use 'gloo' or 'nccl' for multi-GPU
            init_method='env://',
            rank=rank,
            world_size=world_size
        )


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_uvp(rank, world_size, config, console):

    if world_size > 1:
        setup(rank, world_size)

    is_distributed = world_size > 1

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    console.info(f"Running on:  {device}")

    config.device = device

    # Define data transformations
    if config.training_contrastive.padding:
        resize_operation = ResizeAndPad((config.training_contrastive.target_size[0],
                                         config.training_contrastive.target_size[1]))
    else:
        resize_operation = transforms.Resize((config.training_contrastive.target_size[0],
                                              config.training_contrastive.target_size[1]))

    transform_base = [
        resize_operation,
        # RandomHorizontalFlip(),
        RandomRotation(degrees=30),
        # RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
        RandomAffine(degrees=15, translate=(0.1, 0.1)),
        GaussianNoise(std=0.1),
        # RandomResizedCrop((config.training.target_size[0], config.training.target_size[1])),
        # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # RandomGrayscale(p=0.1),
        # RandomPerspective(distortion_scale=0.2, p=0.5),
        # RandomVerticalFlip(p=0.1),
        transforms.ToTensor(),
    ]

    transform_sim = [
        resize_operation,
        # RandomHorizontalFlip(),
        RandomRotation(degrees=30),
        # RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
        RandomAffine(degrees=15, translate=(0.1, 0.1)),
        GaussianNoise(std=0.1),
        # RandomResizedCrop((config.training.target_size[0], config.training.target_size[1])),
        # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # RandomGrayscale(p=0.1),
        # RandomPerspective(distortion_scale=0.2, p=0.5),
        # RandomVerticalFlip(p=0.1),
        transforms.ToTensor(),
    ]

    transform_train = [transforms.Compose(transform_base), transforms.Compose(transform_sim),
                       transforms.Compose(transform_sim), ]

    transform_val = transforms.Compose([
        resize_operation,
        transforms.ToTensor()
        ])

    # Create uvp dataset datasets for training and validation
    train_dataset = UvpDataset(root_dir=config.input_folder_train,
                               csv_file=config.input_csv_train,
                               transform=transform_train,
                               phase=config.phase,
                               gray=config.training_contrastive.gray)

    val_dataset = UvpDataset(root_dir=config.input_folder_test,
                             csv_file=config.input_csv_val,
                             transform=transform_val,
                             phase='test',
                             gray=config.training_contrastive.gray)


    if is_distributed:
        sampler_train = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        sampler_val = None
    else:
        sampler_train = None
        sampler_val = None

    train_loader = DataLoader(train_dataset,
                              batch_size=config.training_contrastive.batch_size,
                              sampler=sampler_train,
                              shuffle=(not is_distributed),
                              num_workers=config.training_contrastive.num_workers)

    val_loader = DataLoader(val_dataset,
                            batch_size=config.training_contrastive.batch_size,
                            shuffle=False,
                            num_workers=config.training_contrastive.num_workers,
                            sampler=sampler_val)

    model = resnext.Model(name=config.training_contrastive.architecture_type, num_classes=train_dataset.num_class,
                          feat_dim=config.training_contrastive.feat_dim,
                          use_norm=config.training_contrastive.use_norm,
                          gray=config.training_contrastive.gray)

    model.to(device)
    # test memory usage
    # console.info(memory_usage(config, model, device))

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    if config.training_contrastive.path_pretrain:
        pth_files = [file for file in os.listdir(config.training_path) if
                     file.endswith('.pth') and file != 'model_weights_best.pth']
        epochs = [int(file.split('_')[-1].split('.')[0]) for file in pth_files]
        latest_epoch = max(epochs)
        latest_pth_file = f"model_weights_epoch_{latest_epoch}.pth"

        saved_weights_file = os.path.join(config.training_path, latest_pth_file)
        state_dict = torch.load(saved_weights_file, map_location=device)

        if world_size > 1:
            new_state_dict = state_dict
        else:
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        console.info("Model loaded from ", saved_weights_file)
        model.load_state_dict(new_state_dict, strict=True)
        model.to(device)
    else:
        latest_epoch = 0

    # Loss criterion and optimizer
    # class_counts = train_dataset.data_frame['label'].value_counts().sort_index().tolist()
    # total_samples = sum(class_counts)
    # class_weights = [total_samples / (train_dataset.num_class * count) for count in class_counts]
    # class_weights_tensor = torch.FloatTensor(class_weights)
    # class_weights_tensor_normalize = class_weights_tensor / class_weights_tensor.sum()

    cls_num_list = train_dataset.data_frame['label'].value_counts().sort_index().tolist()
    class_frequencies = torch.tensor(cls_num_list, dtype=torch.float32)
    class_frequencies = class_frequencies.to(config.device)

    config.cls_num = len(cls_num_list)

    leaf_class_names, super_classes_id, \
    leaf_to_superclass_dict, super_class_names = leaf_class(train_dataset, config)

    prototypes_per_superclass = [1] * config.training_contrastive.superclass_num
    assert len(prototypes_per_superclass) == 20, "We have 20 superclasses"

    if config.training_contrastive.loss == 'proco':
        criterion_ce = LogitAdjust(cls_num_list, device=device)
        criterion_scl = ProCoLoss(contrast_dim=config.training_contrastive.feat_dim,
                                  temperature=config.training_contrastive.temp,
                                  num_classes=train_dataset.num_class,
                                  device=device)

    elif config.training_contrastive.loss == 'amproco':
        criterion_ce = LogitAdjust(cls_num_list, device=device)

        offset = train_dataset.num_class
        super_to_protos = {}  # maps superclass_index -> list of prototype IDs
        for i, (sname, leaf_list) in enumerate(super_classes_id):
            p_i = prototypes_per_superclass[i]
            proto_ids = []
            for _ in range(p_i):
                proto_ids.append(offset)
                offset += 1
            super_to_protos[i] = proto_ids

        root_node_id = offset
        offset += 1

        leaf_path_map = {}
        for i, (sname, leaf_list) in enumerate(super_classes_id):
            proto_ids = super_to_protos[i]
            for leaf in leaf_list:
                path = [root_node_id] + proto_ids + [leaf]
                leaf_path_map[leaf] = path

        num_leaves = train_dataset.num_class
        sum_protos = sum(prototypes_per_superclass)
        num_nodes = num_leaves + sum_protos + 1

        assert (offset - 1) < num_nodes, "All IDs must be in range"

        leaf_node_ids = list(range(train_dataset.num_class))
        proco_loss = ProCoLoss(contrast_dim=config.training_contrastive.feat_dim,
                               temperature=config.training_contrastive.temp,
                               num_classes=num_nodes,
                               device=device)

        criterion_scl = HierarchicalProCoWrapper(proco_loss,
                                                 leaf_node_ids=leaf_node_ids,
                                                 leaf_path_map=leaf_path_map,
                                                 num_nodes=num_nodes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), config.training_contrastive.learning_rate,
                                momentum=config.training_contrastive.momentum,
                                weight_decay=config.training_contrastive.weight_decay)

    # if config.training_contrastive.path_pretrain:
    #     proco_loss.reload_memory()

    ce_loss_all_avg = []
    scl_loss_all_avg = []
    top1_avg = []
    top1_val_avg = []
    best_acc1 = 0.0

    # Training loop
    for epoch in range(latest_epoch, config.training_contrastive.num_epoch):

        if is_distributed and sampler_train is not None:
            sampler_train.set_epoch(epoch)

        adjust_lr(optimizer, epoch, config)

        if epoch < config.training_contrastive.twostage_epoch:
            ce_loss_all, scl_loss_all, top1 = train(epoch, train_loader, model, criterion_ce, criterion_scl, optimizer,
                                                    config, console)
        else:
            if epoch == config.training_contrastive.twostage_epoch:
                superclass_feats = cal_feats(model, train_loader, leaf_to_superclass_dict, config)
                p_star, mixture_params = cal_params(superclass_feats, config.training_contrastive.superclass_num,
                                                    config.training_contrastive.k_max,
                                                    config.training_contrastive.delta_min)

                console.info('super class names   :' + str(super_class_names))
                console.info('P*   :' + str(p_star))

                offset = train_dataset.num_class
                superclass_to_protos = {}
                for i, (sname, leaf_list) in enumerate(super_classes_id):
                    p_i = p_star[i]
                    proto_list = []
                    for comp in range(p_i):
                        proto_list.append(offset)
                        offset += 1
                    superclass_to_protos[i] = proto_list

                root_node_id = offset
                offset += 1
                num_nodes = train_dataset.num_class + sum(p_star) + 1

                leaf_path_map = {}
                for i, (sname, leaf_list) in enumerate(super_classes_id):
                    proto_ids = superclass_to_protos[i]
                    for leaf_id in leaf_list:
                        # path => [root_node_id] + proto_ids + [leaf_id]
                        leaf_path_map[leaf_id] = [root_node_id] + proto_ids + [leaf_id]

                leaf_node_ids = list(range(train_dataset.num_class))

                new_proco_loss = ProCoLoss(contrast_dim=config.training_contrastive.feat_dim,
                                           temperature=config.training_contrastive.temp,
                                           num_classes=num_nodes,
                                           device=device)

                new_criterion_scl = HierarchicalProCoWrapper(
                    proco_loss=new_proco_loss,
                    leaf_node_ids=leaf_node_ids,
                    leaf_path_map=leaf_path_map,
                    num_nodes=num_nodes).to(device)

                for sc_idx in range(config.training_contrastive.superclass_num):
                    p_i = p_star[sc_idx]
                    proto_list = superclass_to_protos[sc_idx]
                    for j in range(p_i):
                        node_id = proto_list[j]
                        (pi_j, mu_j, kappa_j) = mixture_params[sc_idx][j]
                        # mu_j is a numpy array of shape [feature_dim]
                        # ensure it's normalized
                        mu_j = mu_j / (np.linalg.norm(mu_j) + 1e-12)
                        # set them in the Estimator
                        new_proco_loss.estimator.Ave[node_id] = torch.from_numpy(mu_j).to(device)
                        new_proco_loss.estimator.kappa[node_id] = torch.tensor(kappa_j, device=device)
                        # logC can be updated or left to be updated in next iteration (update_kappa).

            ce_loss_all, scl_loss_all, top1 = train(epoch, train_loader, model, criterion_ce, new_criterion_scl,
                                                    optimizer, config, console)

            if epoch == config.training_contrastive.num_epoch - 1:
                console.info('kappa values for superclasses   :' + str(new_proco_loss.estimator.kappa[100:-1]))


        ce_loss_all_avg.append(ce_loss_all.avg)
        scl_loss_all_avg.append(scl_loss_all.avg)
        top1_avg.append(top1.avg)

        plot_loss(ce_loss_all_avg, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path,
                  name='CE_loss.png')
        plot_loss(scl_loss_all_avg, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path,
                  name='SCL_loss.png')
        plot_loss(top1_avg, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path, name='ACC.png')

        if is_distributed:
            dist.barrier()

        if rank != -1:
            acc1, many, med, few, total_labels, all_preds, all_features = validate(train_loader, val_loader, model, criterion_ce, config, console)

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                best_many = many
                best_med = med
                best_few = few
                console.info('Epoch: {:.3f}, Best Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: '
                             '{:.3f}'.format(round(epoch+1), best_acc1, best_many, best_med, best_few))

                # Save the model weights
                saved_weights_best = f'model_weights_best.pth'
                saved_weights_file_best = os.path.join(config.training_path, saved_weights_best)

                console.info(f"Model weights saved to {saved_weights_file_best}")
                torch.save(model.state_dict(), saved_weights_file_best)

            top1_val_avg.append(acc1)
            plot_loss(top1_val_avg, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path,
                      name='ACC_validation.png')

            if epoch % 20 == 0:
                plot_tsne_from_validate(
                    all_features=all_features,
                    total_labels=total_labels,
                    class_to_superclass=leaf_to_superclass_dict,
                    leaf_class_names=leaf_class_names,
                    super_class_names=super_class_names,
                    title_prefix="ValSet",
                    save_dir=os.path.join(config.training_path, 'tsne'),  # e.g. your desired directory
                    epoch=epoch  # e.g. if you're at epoch 20
                )

    if rank != -1:
        # Create a plot of the loss values
        plot_loss(ce_loss_all_avg, num_epoch=(config.training_contrastive.num_epoch - latest_epoch), training_path=config.training_path, name='CE_loss.png')
        plot_loss(scl_loss_all_avg, num_epoch=(config.training_contrastive.num_epoch - latest_epoch), training_path=config.training_path, name='SCL_loss.png')
        plot_loss(top1_avg, num_epoch=(config.training_contrastive.num_epoch - latest_epoch), training_path=config.training_path, name='ACC.png')

        # Save the model's state dictionary to a file
        saved_weights = f'model_weights_epoch_{config.training_contrastive.num_epoch}.pth'
        saved_weights_file = os.path.join(config.training_path, saved_weights)

        torch.save(model.state_dict(), saved_weights_file)

        console.info(f"Final model weights saved to {saved_weights_file}")

    if is_distributed:
        dist.barrier()

    if rank != -1:
        # load best model
        saved_weights_best = f'model_weights_best.pth'
        saved_weights_file_best = os.path.join(config.training_path, saved_weights_best)

        console.info("Best Model loaded from ", saved_weights_file_best)

        state_dict = torch.load(saved_weights_file_best, map_location=device)

        if world_size > 1:
            new_state_dict = state_dict
        else:
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        console.info("Model loaded from ", saved_weights_file_best)
        model.load_state_dict(new_state_dict, strict=True)
        model.to(device)

        test_dataset = UvpDataset(root_dir=config.input_folder_test,
                                  csv_file=config.input_csv_test,
                                  transform=transform_val,
                                  phase='test',
                                  gray=config.training_contrastive.gray)

        test_loader = DataLoader(test_dataset,
                                 batch_size=config.training_contrastive.batch_size,
                                 shuffle=True,
                                 num_workers=config.training_contrastive.num_workers)

        acc1, many, med, few, total_labels, all_preds, all_features = validate(train_loader, test_loader, model, criterion_ce, config, console)

        total_labels = total_labels.cpu().numpy()
        all_preds = all_preds.cpu().numpy()

        report = classification_report(
            total_labels,
            all_preds,
            target_names=train_dataset.class_to_idx,
            digits=6,
        )

        conf_mtx = confusion_matrix(
            total_labels,
            all_preds,
        )

        df = report_to_df(report)
        report_filename = os.path.join(config.training_path, 'report_evaluation.csv')
        df.to_csv(report_filename)

        df = pd.DataFrame(conf_mtx)
        conf_mtx_filename = os.path.join(config.training_path, 'conf_matrix_evaluation.csv')
        df.to_csv(conf_mtx_filename)

        console.info('************* Evaluation Report *************')
        console.info(report)
        console.save_log(config.training_path)

        console.info('************* Plot T-sne *************')

        plot_tsne_from_validate(
            all_features=all_features,
            total_labels=total_labels,
            class_to_superclass=leaf_to_superclass_dict,
            leaf_class_names=leaf_class_names,
            super_class_names=super_class_names,
            title_prefix="ValSet",
            save_dir=os.path.join(config.training_path, 'tsne'),  # e.g. your desired directory
            epoch=config.training_contrastive.num_epoch
        )


def train_cifar(rank, world_size, config, console):

    if world_size > 1:
        setup(rank, world_size)

    is_distributed = world_size > 1

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    console.info(f"Running on:  {device}")

    config.device = device

    if config.training_contrastive.num_epoch == 200:
        config.training_contrastive.schedule = [160, 180]
        config.training_contrastive.warmup_epoch = 5
    elif config.training_contrastive.num_epoch == 400:
        config.training_contrastive.schedule = [360, 380]
        config.training_contrastive.warmup_epoch = 10
    else:
        config.training_contrastive.schedule = [config.training_contrastive.num_epoch * 0.8,
                                                config.training_contrastive.num_epoch * 0.9]
        config.training_contrastive.warmup_epoch = 5 * config.training_contrastive.num_epoch // 200

    # Define data transformations
    augmentation_regular = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]

    augmentation_sim_cifar = [
        transforms.RandomResizedCrop(size=32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    transform_train = [transforms.Compose(augmentation_regular),
                       transforms.Compose(augmentation_sim_cifar),
                       transforms.Compose(augmentation_sim_cifar)]

    transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if config.training_contrastive.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root=config.input_folder_train, imb_type='exp',
                                          imb_factor=config.training_contrastive.im_factor,
                                          rand_number=0,
                                          train=True,
                                          download=True,
                                          transform=transform_train)
        val_dataset = datasets.CIFAR100(
                root=config.input_folder_train,
                train=False,
                download=True,
                transform=transform_val)
    else:
        raise ValueError('Unknown dataset')

    console.info(f'===> Training data length {len(train_dataset)}')
    console.info(f'===> Validation data length {len(val_dataset)}')

    train_dataset.num_class = 100

    if is_distributed:
        sampler_train = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        # sampler_val = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        sampler_val = None
    else:
        sampler_train = None
        sampler_val = None

    train_loader = DataLoader(train_dataset,
                              batch_size=config.training_contrastive.batch_size,
                              sampler=sampler_train,
                              shuffle=(not is_distributed),
                              num_workers=config.training_contrastive.num_workers)

    val_loader = DataLoader(
        val_dataset, batch_size=config.training_contrastive.batch_size, shuffle=False,
        num_workers=config.training_contrastive.num_workers, pin_memory=True, sampler=sampler_val)

    if config.training_contrastive.architecture_type == 'resnet32':
        model = resnet_cifar.Model(name=config.training_contrastive.architecture_type,
                                   num_classes=train_dataset.num_class,
                                   feat_dim=config.training_contrastive.feat_dim,
                                   use_norm=config.training_contrastive.use_norm)
    else:
        raise NotImplementedError("only select resnet32 architecture for cifar datasets!")

    model.to(device)

    # test memory usage
    # console.info(memory_usage(config, model, device))

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    if config.training_contrastive.path_pretrain:
        pth_files = [file for file in os.listdir(config.training_path) if
                     file.endswith('.pth') and file != 'model_weights_best.pth']
        epochs = [int(file.split('_')[-1].split('.')[0]) for file in pth_files]
        latest_epoch = max(epochs)
        latest_pth_file = f"model_weights_epoch_{latest_epoch}.pth"

        saved_weights_file = os.path.join(config.training_path, latest_pth_file)
        state_dict = torch.load(saved_weights_file, map_location=device)

        if world_size > 1:
            new_state_dict = state_dict
        else:
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        console.info("Model loaded from ", saved_weights_file)
        model.load_state_dict(new_state_dict, strict=True)
        model.to(device)
    else:
        latest_epoch = 0

    # Loss criterion and optimizer
    cls_num_list = train_dataset.get_cls_num_list()
    class_frequencies = torch.tensor(cls_num_list, dtype=torch.float32)
    class_frequencies = class_frequencies.to(config.device)

    config.cls_num = len(cls_num_list)

    leaf_class_names, super_classes_id, \
    leaf_to_superclass_dict, super_class_names = leaf_class(train_dataset, config)

    prototypes_per_superclass = [1] * config.training_contrastive.superclass_num
    assert len(prototypes_per_superclass) == 20, "We have 20 superclasses"

    if config.training_contrastive.loss == 'proco':
        criterion_ce = LogitAdjust(cls_num_list, device=device)
        criterion_scl = ProCoLoss(contrast_dim=config.training_contrastive.feat_dim,
                                  temperature=config.training_contrastive.temp,
                                  num_classes=train_dataset.num_class,
                                  device=device)

    elif config.training_contrastive.loss == 'amproco':
        criterion_ce = LogitAdjust(cls_num_list, device=device)

        offset = train_dataset.num_class
        super_to_protos = {}  # maps superclass_index -> list of prototype IDs
        for i, (sname, leaf_list) in enumerate(super_classes_id):
            p_i = prototypes_per_superclass[i]
            proto_ids = []
            for _ in range(p_i):
                proto_ids.append(offset)
                offset += 1
            super_to_protos[i] = proto_ids

        root_node_id = offset
        offset += 1

        leaf_path_map = {}
        for i, (sname, leaf_list) in enumerate(super_classes_id):
            proto_ids = super_to_protos[i]
            for leaf in leaf_list:
                path = [root_node_id] + proto_ids + [leaf]
                leaf_path_map[leaf] = path

        num_leaves = train_dataset.num_class
        sum_protos = sum(prototypes_per_superclass)
        num_nodes = num_leaves + sum_protos + 1

        assert (offset - 1) < num_nodes, "All IDs must be in range"

        leaf_node_ids = list(range(train_dataset.num_class))
        proco_loss = ProCoLoss(contrast_dim=config.training_contrastive.feat_dim,
                               temperature=config.training_contrastive.temp,
                               num_classes=num_nodes,
                               device=device)

        criterion_scl = HierarchicalProCoWrapper(proco_loss,
                                                 leaf_node_ids=leaf_node_ids,
                                                 leaf_path_map=leaf_path_map,
                                                 num_nodes=num_nodes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), config.training_contrastive.learning_rate,
                                momentum=config.training_contrastive.momentum,
                                weight_decay=config.training_contrastive.weight_decay)

    # if config.training_contrastive.path_pretrain:
    #     proco_loss.reload_memory()

    ce_loss_all_avg = []
    scl_loss_all_avg = []
    top1_avg = []
    top1_val_avg = []
    best_acc1 = 0.0

    # Training loop
    for epoch in range(latest_epoch, config.training_contrastive.num_epoch):

        if is_distributed and sampler_train is not None:
            sampler_train.set_epoch(epoch)

        adjust_lr(optimizer, epoch, config)

        if epoch < config.training_contrastive.twostage_epoch:
            ce_loss_all, scl_loss_all, top1 = train(epoch, train_loader, model, criterion_ce, criterion_scl, optimizer,
                                                    config, console)
        else:
            if epoch == config.training_contrastive.twostage_epoch:
                superclass_feats = cal_feats(model, train_loader, leaf_to_superclass_dict, config)
                p_star, mixture_params = cal_params(superclass_feats, config.training_contrastive.superclass_num,
                                                    config.training_contrastive.k_max,
                                                    config.training_contrastive.delta_min)

                console.info('super class names   :' + str(super_class_names))
                console.info('P*   :' + str(p_star))

                offset = train_dataset.num_class
                superclass_to_protos = {}
                for i, (sname, leaf_list) in enumerate(super_classes_id):
                    p_i = p_star[i]
                    proto_list = []
                    for comp in range(p_i):
                        proto_list.append(offset)
                        offset += 1
                    superclass_to_protos[i] = proto_list

                root_node_id = offset
                offset += 1
                num_nodes = train_dataset.num_class + sum(p_star) + 1

                leaf_path_map = {}
                for i, (sname, leaf_list) in enumerate(super_classes_id):
                    proto_ids = superclass_to_protos[i]
                    for leaf_id in leaf_list:
                        # path => [root_node_id] + proto_ids + [leaf_id]
                        leaf_path_map[leaf_id] = [root_node_id] + proto_ids + [leaf_id]

                leaf_node_ids = list(range(train_dataset.num_class))

                new_proco_loss = ProCoLoss(contrast_dim=config.training_contrastive.feat_dim,
                                           temperature=config.training_contrastive.temp,
                                           num_classes=num_nodes,
                                           device=device)

                new_criterion_scl = HierarchicalProCoWrapper(
                    proco_loss=new_proco_loss,
                    leaf_node_ids=leaf_node_ids,
                    leaf_path_map=leaf_path_map,
                    num_nodes=num_nodes).to(device)

                for sc_idx in range(config.training_contrastive.superclass_num):
                    p_i = p_star[sc_idx]
                    proto_list = superclass_to_protos[sc_idx]
                    for j in range(p_i):
                        node_id = proto_list[j]
                        (pi_j, mu_j, kappa_j) = mixture_params[sc_idx][j]
                        # mu_j is a numpy array of shape [feature_dim]
                        # ensure it's normalized
                        mu_j = mu_j / (np.linalg.norm(mu_j) + 1e-12)
                        # set them in the Estimator
                        new_proco_loss.estimator.Ave[node_id] = torch.from_numpy(mu_j).to(device)
                        new_proco_loss.estimator.kappa[node_id] = torch.tensor(kappa_j, device=device)
                        # logC can be updated or left to be updated in next iteration (update_kappa).

            ce_loss_all, scl_loss_all, top1 = train(epoch, train_loader, model, criterion_ce, new_criterion_scl,
                                                    optimizer, config, console)

            if epoch == config.training_contrastive.num_epoch - 1:
                console.info('kappa values for superclasses   :' + str(new_proco_loss.estimator.kappa[100:-1]))

        ce_loss_all_avg.append(ce_loss_all.avg)
        scl_loss_all_avg.append(scl_loss_all.avg)
        top1_avg.append(top1.avg)

        plot_loss(ce_loss_all_avg, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path,
                  name='CE_loss.png')
        plot_loss(scl_loss_all_avg, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path,
                  name='SCL_loss.png')
        plot_loss(top1_avg, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path, name='ACC.png')

        if is_distributed:
            dist.barrier()

        if rank != -1:
            acc1, many, med, few, total_labels, all_preds, all_features = validate(train_loader, val_loader, model, criterion_ce, config, console)

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                best_many = many
                best_med = med
                best_few = few
                console.info('Epoch: {:.3f}, Best Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: '
                             '{:.3f}'.format(round(epoch+1), best_acc1, best_many, best_med, best_few))

                # Save the model weights
                saved_weights_best = f'model_weights_best.pth'
                saved_weights_file_best = os.path.join(config.training_path, saved_weights_best)

                console.info(f"Model weights saved to {saved_weights_file_best}")
                torch.save(model.state_dict(), saved_weights_file_best)

            top1_val_avg.append(acc1)
            plot_loss(top1_val_avg, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path,
                      name='ACC_validation.png')

            # if epoch % 20 == 0:
            #     plot_tsne_from_validate(
            #         all_features=all_features,
            #         total_labels=total_labels,
            #         class_to_superclass=leaf_to_superclass_dict,
            #         leaf_class_names=leaf_class_names,
            #         super_class_names=super_class_names,
            #         title_prefix="ValSet",
            #         save_dir=os.path.join(config.training_path, 'tsne'),  # e.g. your desired directory
            #         epoch=epoch  # e.g. if you're at epoch 20
            #     )

    if rank != -1:
        # Create a plot of the loss values
        plot_loss(ce_loss_all_avg, num_epoch=(config.training_contrastive.num_epoch - latest_epoch), training_path=config.training_path, name='CE_loss.png')
        plot_loss(scl_loss_all_avg, num_epoch=(config.training_contrastive.num_epoch - latest_epoch), training_path=config.training_path, name='SCL_loss.png')
        plot_loss(top1_avg, num_epoch=(config.training_contrastive.num_epoch - latest_epoch), training_path=config.training_path, name='ACC.png')

        # Save the model's state dictionary to a file
        saved_weights = f'model_weights_epoch_{config.training_contrastive.num_epoch}.pth'
        saved_weights_file = os.path.join(config.training_path, saved_weights)

        torch.save(model.state_dict(), saved_weights_file)

        console.info(f"Final model weights saved to {saved_weights_file}")

    if is_distributed:
        dist.barrier()

    if rank != -1:
        # load best model
        saved_weights_best = f'model_weights_best.pth'
        saved_weights_file_best = os.path.join(config.training_path, saved_weights_best)

        console.info("Best Model loaded from ", saved_weights_file_best)

        state_dict = torch.load(saved_weights_file_best, map_location=device)

        if world_size > 1:
            new_state_dict = state_dict
        else:
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        console.info("Model loaded from ", saved_weights_file_best)
        model.load_state_dict(new_state_dict, strict=True)
        model.to(device)

        test_loader = val_loader

        acc1, many, med, few, total_labels, all_preds, all_features = validate(train_loader, test_loader, model, criterion_ce, config, console)

        total_labels = total_labels.cpu().numpy()
        all_preds = all_preds.cpu().numpy()

        report = classification_report(
            total_labels,
            all_preds,
            digits=6,
        )

        conf_mtx = confusion_matrix(
            total_labels,
            all_preds,
        )

        df = report_to_df(report)
        report_filename = os.path.join(config.training_path, 'report_evaluation.csv')
        df.to_csv(report_filename)

        df = pd.DataFrame(conf_mtx)
        conf_mtx_filename = os.path.join(config.training_path, 'conf_matrix_evaluation.csv')
        df.to_csv(conf_mtx_filename)

        console.info('************* Evaluation Report *************')
        console.info(report)
        console.save_log(config.training_path)

        console.info('************* Plot T-sne *************')

        plot_tsne_from_validate(
            all_features=all_features,
            total_labels=total_labels,
            class_to_superclass=leaf_to_superclass_dict,
            leaf_class_names=leaf_class_names,
            super_class_names=super_class_names,
            title_prefix="ValSet",
            save_dir=os.path.join(config.training_path, 'tsne'),  # e.g. your desired directory
            epoch=epoch  # e.g. if you're at epoch 20
        )


def train(epoch, train_loader, model, criterion_ce, criterion_scl, optimizer, config, console):
    model.train()

    if hasattr(criterion_scl, "_hook_before_epoch"):
        criterion_scl._hook_before_epoch()

    batch_time = AverageMeter('Time', ':6.3f')
    ce_loss_all = AverageMeter('CE_Loss', ':.4e')
    scl_loss_all = AverageMeter('SCL_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    end = time.time()
    for batch_idx, data in enumerate(train_loader):
        if len(data) == 3:
            images, labels, _ = data
        elif len(data) == 2:
            images, labels = data
        else:
            raise ValueError("Unexpected number of elements returned by train_loader.")

        batch_size = labels.shape[0]
        labels = labels.to(config.device)

        mini_batch_size = batch_size // config.training_contrastive.accumulation_steps

        images_0_mini_batches = torch.split(images[0], mini_batch_size)
        images_1_mini_batches = torch.split(images[1], mini_batch_size)
        images_2_mini_batches = torch.split(images[2], mini_batch_size)
        labels_mini_batches = torch.split(labels, mini_batch_size)

        optimizer.zero_grad()

        aggregated_logits = []

        for i in range(len(images_0_mini_batches)):
            mini_images = torch.cat([images_0_mini_batches[i], images_1_mini_batches[i], images_2_mini_batches[i]],
                                    dim=0)
            mini_labels = labels_mini_batches[i]

            mini_images, mini_labels = mini_images.to(config.device), mini_labels.to(config.device)

            feat_mlp, ce_logits, _ = model(mini_images)
            _, f2, f3 = torch.split(feat_mlp, [mini_batch_size, mini_batch_size, mini_batch_size], dim=0)
            ce_logits, _, __ = torch.split(ce_logits, [mini_batch_size, mini_batch_size, mini_batch_size], dim=0)

            contrast_logits1 = criterion_scl(f2, mini_labels)
            contrast_logits2 = criterion_scl(f3, mini_labels)

            contrast_logits1, contrast_logits2 = contrast_logits1.to(config.device), contrast_logits2.to(config.device)

            contrast_logits = (contrast_logits1 + contrast_logits2) / 2

            scl_loss = (F.cross_entropy(contrast_logits1, mini_labels) + F.cross_entropy(contrast_logits2, mini_labels)) / 2
            ce_loss = criterion_ce(ce_logits, mini_labels)

            alpha = 1
            if epoch > 200:
                lambda_ = 0
            else:
                lambda_ = 1
            logits = ce_logits + alpha * contrast_logits
            loss = lambda_ * ce_loss + alpha * scl_loss

            # Accumulate gradients
            loss.backward()
            aggregated_logits.append(logits)

        optimizer.step()
        aggregated_logits = torch.cat(aggregated_logits, dim=0)
        aggregated_logits = aggregated_logits.to(config.device)

        ce_loss_all.update(ce_loss.item(), batch_size)
        scl_loss_all.update(scl_loss.item(), batch_size)

        acc1 = accuracy(aggregated_logits, labels, topk=(1,))
        top1.update(acc1[0].item(), batch_size)

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # # for debug
        # from tools.image import save_img
        # save_img(images, batch_idx, epoch, training_path/"augmented")

        # if batch_idx % 20 == 0:
        #     output = ('Epoch: [{0}][{1}/{2}] \t'
        #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #               'CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
        #               'SCL_Loss {scl_loss.val:.4f} ({scl_loss.avg:.4f})\t'
        #               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #         epoch, batch_idx, len(train_loader), batch_time=batch_time,
        #         ce_loss=ce_loss_all, scl_loss=scl_loss_all, top1=top1, ))  # TODO
        #     print(output)

    console.info(f"CE loss train [{epoch + 1}/{config.training_contrastive.num_epoch}] - Loss: {ce_loss_all.avg:.4f} ")
    console.info(
        f"SCL loss train [{epoch + 1}/{config.training_contrastive.num_epoch}] - Loss: {scl_loss_all.avg:.4f} ")
    console.info(f"acc train top1 [{epoch + 1}/{config.training_contrastive.num_epoch}] - Acc: {top1.avg:.4f} ")

    return ce_loss_all, scl_loss_all, top1


def validate(train_loader, val_loader, model, criterion_ce, config, console):

    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    ce_loss_all = AverageMeter('CE_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    total_logits = torch.empty((0, train_loader.dataset.num_class)).to(config.device)
    total_labels = torch.empty(0, dtype=torch.long).to(config.device)
    all_features = []

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            if len(data) == 3:
                images, labels, img_names = data
            elif len(data) == 2:
                images, labels = data
            else:
                raise ValueError("Unexpected number of elements returned by train_loader.")

            images, labels = images.to(config.device), labels.to(config.device)

            feat_mlp, ce_logits, _ = model(images)
            logits = ce_logits

            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, labels))
            all_features.append(feat_mlp.cpu().numpy())

            batch_time.update(time.time() - end)

            probs, preds = F.softmax(logits, dim=1).max(dim=1)
            save_image = False
            if save_image:
                for i in range(len(preds)):
                    int_label = preds[i].item()
                    string_label = val_loader.dataset.get_string_label(int_label)
                    image_name = img_names[i]
                    image_path = os.path.join(config.training_path, 'output/', string_label,
                                              image_name.replace('output/', ''))

                    if not os.path.exists(os.path.dirname(image_path)):
                        os.makedirs(os.path.dirname(image_path))

                    input_path = os.path.join(val_loader.dataset.root_dir, image_name)
                    shutil.copy(input_path, image_path)


        ce_loss = criterion_ce(total_logits, total_labels)
        acc1 = accuracy(total_logits, total_labels, topk=(1,))

        ce_loss_all.update(ce_loss.item(), 1)
        top1.update(acc1[0].item(), 1)

        all_probs, all_preds = F.softmax(total_logits, dim=1).max(dim=1)
        many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(all_preds, total_labels, train_loader,
                                                                acc_per_cls=False)
        acc1 = top1.avg
        many = many_acc_top1 * 100
        med = median_acc_top1 * 100
        few = low_acc_top1 * 100
        console.info(
            'Validation: Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(acc1, many, med, few))

        all_features = np.concatenate(all_features, axis=0)

        return acc1, many, med, few, total_labels, all_preds, all_features


def leaf_class(train_dataset, config):

    if config.training_contrastive.dataset=='uvp':
        superclass = train_dataset.UVP_SUPERCLASSES
    else:
        superclass = CIFAR100_SUPERCLASSES

    train_class2idx = train_dataset.class_to_idx
    super_classes_id = []
    for superclass_name, leaf_names in superclass:
        leaf_ids = [train_class2idx[leaf_name] for leaf_name in leaf_names]
        super_classes_id.append((superclass_name, leaf_ids))

    leaf_to_superclass_dict = {}
    super_class_names = []
    for sup_id, (sup_name, leaf_ids) in enumerate(super_classes_id):
        super_class_names.append(sup_name)
        for leaf_id in leaf_ids:
            leaf_to_superclass_dict[leaf_id] = sup_id

    leaf_class_names = [name for name, idx in train_dataset.class_to_idx.items()]

    return leaf_class_names, super_classes_id, leaf_to_superclass_dict, super_class_names


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def adjust_lr(optimizer, epoch, config):
    """Decay the learning rate based on schedule"""
    lr = config.training_contrastive.learning_rate
    if epoch < config.training_contrastive.warmup_epoch:
        lr = lr / config.training_contrastive.warmup_epoch * (epoch + 1)
    elif config.training_contrastive.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - config.training_contrastive.warmup_epoch + 1) /
                                   (config.training_contrastive.num_epoch - config.training_contrastive.warmup_epoch + 1)))
    else:  # stepwise lr schedule
        for milestone in config.training_contrastive.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cal_feats(model, train_loader, leaf_to_superclass_dict, config):
    superclass_feats = [[] for _ in range(20)]
    for i, data in enumerate(train_loader):
        if len(data) == 3:
            images, leaf_label, img_names = data
        elif len(data) == 2:
            images, leaf_label = data
        images, leaf_label = images[0].to(config.device), leaf_label.to(config.device)
        with torch.no_grad():
            z, ce_logits, _ = model(images)
            z = F.normalize(z, p=2, dim=1)  # ensure unit sphere if needed
        leaf_label_array = leaf_label.cpu().numpy()
        for i in range(len(leaf_label)):
            sc_idx = leaf_to_superclass_dict[leaf_label_array[i]]  # e.g. a function returning [0..19]
            superclass_feats[sc_idx].append(z[i].cpu().numpy())

    return superclass_feats


def cal_params(superclass_feats, superclass_num, k_max=5, delta_min=100):
    p_star = []
    mixture_params = {}  # store (pi_j, mu_j, kappa_j) for each j in [1.. best_k]
    for sc_idx in range(superclass_num):
        feats_sc = np.array(superclass_feats[sc_idx])  # shape [N_sc, feat_dim]
        best_k, best_params = find_best_vmf_mixture_bic(feats_sc, k_max=k_max, delta_min=delta_min)
        p_star.append(best_k)
        mixture_params[sc_idx] = best_params

    return p_star, mixture_params


def find_best_vmf_mixture_bic(feats_sc, k_max=5, delta_min=100):
    """
    feats_sc: shape [N_sc, feat_dim]
    returns:
        best_k: the number of prototypes with the minimal BIC
        best_params: a list of (pi_j, mu_j, kappa_j) for j=1..best_k
    """
    min_improvement = delta_min
    best_k = 1
    best_bic = float('inf')
    best_params = None

    N_sc, dim = feats_sc.shape

    prev_bic = None
    prev_params = None

    for k in range(1, k_max + 1):
        # 1) Fit a mixture-of-vMF with k components to feats_sc
        mixture_params_k = fit_vmf_mixture(feats_sc, k)

        # 2) compute log-likelihood: sum_{i=1..N_sc} log( sum_{j=1..k} pi_j * vmf_pdf(...) )
        logL = 0.0
        for i in range(N_sc):
            x = feats_sc[i]
            pdf_sum = 0.0
            for (pi_j, mu_j, kappa_j) in mixture_params_k:
                pdf_sum += pi_j * vmf_pdf(x, mu_j, kappa_j)
            logL += np.log(pdf_sum + 1e-20)

        # 3) compute param count
        #   each component: (dim-1) for mu, 1 for kappa, total k comps => k*(dim)
        #   plus (k-1) for pi_j. So total = k*(dim) + (k-1).
        #   or you can do k*(dim -1) + k + (k-1), etc.
        #   You can approximate it as:
        param_count = k * (dim) + (k - 1)

        # 4) BIC = -2 * logL + param_count * ln(N_sc)
        bic_value = -2.0 * logL + param_count * np.log(N_sc)

        if k == 1:
            best_k = 1
            best_params = mixture_params_k
            best_bic = bic_value
            prev_bic = bic_value
            prev_params = mixture_params_k
        else:
            delta_bic = prev_bic - bic_value
            print('in k = ' + str(k) + '   , the delta is:  ' + str(delta_bic))

            if delta_bic < min_improvement:
                best_k = k - 1
                best_params = prev_params
                best_bic = prev_bic
                break
            else:
                best_k = k
                best_params = mixture_params_k
                best_bic = bic_value
                prev_bic = bic_value
                prev_params = mixture_params_k

        # if bic_value < best_bic:
        #     best_bic = bic_value
        #     best_k = k
        #     best_params = mixture_params_k

    return best_k, best_params


def fit_vmf_mixture(feats_sc, k, max_iter=100):
    """
    feats_sc: shape [N, dim]
    returns list of (pi_j, mu_j, kappa_j) for j=1..k
    """
    N, dim = feats_sc.shape

    # 1) Initialize pi_j, mu_j, kappa_j
    pi = np.ones(k) / k
    mu = np.random.randn(k, dim)
    mu = mu / np.linalg.norm(mu, axis=1, keepdims=True)  # normalize
    kappa = np.ones(k) * dim  # or random init

    # 2) EM loop
    for _ in range(max_iter):
        # E-step: compute responsibilities
        # shape: R[i, j] = pi_j * vMF_pdf(x_i, mu_j, kappa_j)
        R = np.zeros((N, k))

        for j in range(k):
            for i in range(N):
                R[i, j] = pi[j] * vmf_pdf(feats_sc[i], mu[j], kappa[j])
        R_sum = R.sum(axis=1, keepdims=True) + 1e-30
        R /= R_sum  # responsibilities

        # M-step: update pi_j
        Nj = R.sum(axis=0)  # shape [k]
        pi = Nj / N

        # update mu_j, kappa_j
        # Weighted average of the x_i
        for j in range(k):
            # compute the weighted sum
            weighted_sum = np.zeros(dim)
            for i in range(N):
                weighted_sum += R[i, j] * feats_sc[i]
            # normalization
            norm = np.linalg.norm(weighted_sum)
            if norm < 1e-8:
                # degenerate, re-init or keep
                continue
            new_mu = weighted_sum / norm
            # compute new kappa
            # for vMF, we can approximate kappa via:
            #   Rbar = norm / Nj[j]
            #   kappa_j ~ (Rbar * (dim - Rbar^2)) / (1 - Rbar^2)
            Rbar = norm / Nj[j]
            if Rbar < 1e-6:
                # degenerate
                continue
            new_kappa = (Rbar * (dim - Rbar ** 2)) / (1 - Rbar ** 2 + 1e-12)

            mu[j] = new_mu
            kappa[j] = new_kappa

    # return final
    mixture_params_k = []
    for j in range(k):
        mixture_params_k.append((pi[j], mu[j], kappa[j]))
    return mixture_params_k


def vmf_pdf(x, mu, kappa):
    """
    x, mu: numpy arrays of shape [dim], both assumed unit norm.
    kappa: float
    returns the PDF value as a float.
    """
    dotval = np.dot(x, mu)  # x, mu in R^dim
    # log_val = kappa * dotval - logC_p(kappa, len(x))
    log_val = kappa * dotval + logC_p(kappa, len(x))
    return np.exp(log_val)


def logC_p(kappa, dim):
    """
    Approximate or compute log of the normalization constant C_d(kappa).
    For large kappa, or dimension not too big, you can do a piecewise approach.
    Or call SciPy if available.
    """
    # If you have scip.special.ive, you can do:
    #   val = ive(dim/2 - 1, kappa)  # i_{nu}(kappa)
    #   logC = np.log(val) + kappa - (dim/2 - 1)*np.log(kappa+1e-12)
    # Return that. Example:
    import math
    from scipy.special import ive

    if kappa < 1e-8:
        # near zero, logC_p ~ -log(Surface of sphere), roughly
        # e.g. log((2*pi)^(d/2) / Gamma(d/2)) ...
        # For simplicity, return a constant. It's not critical for small kappa.
        return (dim/2)*math.log(2*math.pi)  # crude
    val = ive(dim/2 - 1, kappa)
    val = max(val, 1e-300)
    logC = math.log(val) + kappa - (dim/2 - 1)*math.log(kappa+1e-12)
    return logC
