# From: https://github.com/kaidic/LDAM-DRW
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.labels = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            if self.train:
                sample1 = self.transform[0](img)
                sample2 = self.transform[1](img)
                sample3 = self.transform[2](img) 
                return [sample1, sample2, sample3], target
            else:
                return self.transform(img), target

        if self.target_transform is not None:
            target = self.target_transform(target)

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


CIFAR100_SUPERCLASSES = [
    ('aquatic_mammals', ['beaver', 'dolphin', 'otter', 'seal', 'whale']),
    ('fish', ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']),
    ('flowers', ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']),
    ('food_containers', ['bottle', 'bowl', 'can', 'cup', 'plate']),
    ('fruit_and_vegetables', ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']),
    ('household_electrical_devices', ['clock', 'keyboard', 'lamp', 'telephone', 'television']),
    ('household_furniture', ['bed', 'chair', 'couch', 'table', 'wardrobe']),
    ('insects', ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']),
    ('large_carnivores', ['bear', 'leopard', 'lion', 'tiger', 'wolf']),
    ('large_man-made_outdoor_things', ['bridge', 'castle', 'house', 'road', 'skyscraper']),
    ('large_natural_outdoor_scenes', ['cloud', 'forest', 'mountain', 'plain', 'sea']),
    ('large_omnivores_and_herbivores', ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo']),
    ('medium-sized_mammals', ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']),
    ('non-insect_invertebrates', ['crab', 'lobster', 'snail', 'spider', 'worm']),
    ('people', ['baby', 'boy', 'girl', 'man', 'woman']),
    ('reptiles', ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']),
    ('small_mammals', ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']),
    ('trees', ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']),
    ('vehicles_1', ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train']),
    ('vehicles_2', ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']),
]

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECIFAR100(root='/DATACENTER/3/zjg/cifar', train=True,
                                 download=True, transform=transform)
