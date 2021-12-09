import pickle
import sys

import numpy as np
import paddle
import paddle.vision.datasets as datasets  # 代替了torchvision.datasets
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class BatchData(paddle.io.Dataset):
    def __init__(self, images, labels, input_transform=None):
        self.images = images
        self.labels = labels
        self.input_transform = input_transform

    def __getitem__(self, index):
        image = self.images[index]
        image = Image.fromarray(np.uint8(image))
        label = self.labels[index]
        if self.input_transform is not None:
            image = self.input_transform(image)
        # label = paddle.LongTensor([label])
        return image, label

    def __len__(self):
        return len(self.images)


class CIFAR10_IncrementalDataset:
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self):
        self.root = './datasets/'
        downloaded_list_train = self.train_list
        downloaded_list_test = self.test_list

        self.train_data = []
        self.train_targets = []
        self.test_data = []
        self.test_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list_train:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.train_data.append(entry['data'])
                self.train_targets.extend(entry['labels'])

        self.train_data = paddle.reshape(np.vstack(self.train_data), [-1, 3, 32, 32])  # 修改
        for file_name, checksum in downloaded_list_test:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.test_data.append(entry['data'])
                self.test_targets.extend(entry['labels'])

        self.test_data = paddle.reshape(np.vstack(self.test_data), [-1, 3, 32, 32])  # 修改
        self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

        self.train_groups, self.test_groups, self.val_groups = self.initialize()
        self.current_step = 0
        self.batch_num = 2

    def initialize(self):
        train_groups = [[], [], [], [], []]
        for train_data, train_label in zip(self.train_data, self.train_targets):
            if train_label < 8:
                train_groups[0].append((train_data, train_label))
            # elif train_label<4:
            #     train_groups[1].append((train_data, train_label))
            # elif train_label<6:
            #     train_groups[2].append((train_data, train_label))
            # elif train_label<8:
            #     train_groups[3].append((train_data, train_label))
            elif train_label < 10:
                train_groups[1].append((train_data, train_label))

        test_groups = [[], [], [], [], []]
        for test_data, test_label in zip(self.test_data, self.test_targets):
            if test_label < 8:
                test_groups[0].append((test_data, test_label))
            # elif test_label < 4:
            #     test_groups[1].append((test_data, test_label))
            # elif test_label < 6:
            #     test_groups[2].append((test_data, test_label))
            # elif test_label < 8:
            #     test_groups[3].append((test_data, test_label))
            elif test_label < 10:
                test_groups[1].append((test_data, test_label))

        val_groups = [[], [], [], [], []]
        # for step in range(2):
        #     old_classes_propotion = step/(step+1)
        #     new_classes_propotion = 1/(step+1)
        #     num_of_old_classes = int(2000*old_classes_propotion)
        #     num_of_new_classes = int(2000*new_classes_propotion)
        #
        #     if step>=1:
        #         old_classes = []
        #         for i in range(step):
        #             old_classes.extend(test_groups[i])
        #         assert (len(old_classes)==2000*step)
        #         random.shuffle(old_classes)
        #         val_groups[step].extend(old_classes[:num_of_old_classes])
        #
        #     random.shuffle(test_groups[step])
        #     val_groups[step].extend(test_groups[step][:num_of_new_classes])
        #     assert (len(val_groups[step])==2000 or len(val_groups[step])==1999)
        for test_data, test_label in zip(self.test_data, self.test_targets):
            val_groups[1].append((test_data, test_label))

        # for step in range(2):
        #     old_classes_propotion = step/(step+1)
        #     new_classes_propotion = 1/(step+1)
        #     num_of_old_classes = int(2000*old_classes_propotion)
        #     num_of_new_classes = int(2000*new_classes_propotion)
        #
        #     if step>=1:
        #         old_classes = []
        #         for i in range(step):
        #             old_classes.extend(test_groups[i])
        #         assert (len(old_classes)==2000*step)
        #         random.shuffle(old_classes)
        #         val_groups[step].extend(old_classes[:num_of_old_classes])
        #
        #     random.shuffle(test_groups[step])
        #     val_groups[step].extend(test_groups[step][:num_of_new_classes])
        #     assert (len(val_groups[step])==2000 or len(val_groups[step])==1999)
        return train_groups, test_groups, val_groups

    def getNextClasses(self, step_b):
        return self.train_groups[step_b], self.val_groups[step_b]


class CIFAR10_:
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self):
        self.root = './datasets/'
        downloaded_list_train = self.train_list
        downloaded_list_test = self.test_list

        self.train_data = []
        self.train_targets = []
        self.test_data = []
        self.test_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list_train:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.train_data.append(entry['data'])
                self.train_targets.extend(entry['labels'])
        self.train_data = paddle.reshape(np.vstack(self.train_data), [-1, 3, 32, 32])
        self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

        for file_name, checksum in downloaded_list_test:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.test_data.append(entry['data'])
                self.test_targets.extend(entry['labels'])

        self.test_data = paddle.reshape(np.vstack(self.test_data), [-1, 3, 32, 32])
        self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

        self.train_groups, self.test_groups = self.initialize()
        self.current_step = 0
        self.batch_num = 5

    def initialize(self):
        train_groups = [[], [], [], [], []]
        for train_data, train_label in zip(self.train_data, self.train_targets):
            if train_label == 9 or train_label == 8:
                train_groups[0].append((train_data, train_label))
            # elif train_label < 4:
            #     train_groups[1].append((train_data, train_label))
            # elif train_label < 6:
            #     train_groups[2].append((train_data, train_label))
            # elif train_label < 8:
            #     train_groups[3].append((train_data, train_label))
            # elif train_label < 10:
            #     train_groups[4].append((train_data, train_label))

        test_groups = [[], [], [], [], []]
        for test_data, test_label in zip(self.test_data, self.test_targets):
            # if test_label ==8 or test_label==9:
            test_groups[0].append((test_data, test_label))
            # elif test_label < 4:
            #     test_groups[1].append((test_data, test_label))
            # elif test_label < 6:
            #     test_groups[2].append((test_data, test_label))
            # elif test_label < 8:
            #     test_groups[3].append((test_data, test_label))
            # elif test_label < 10:
            #     test_groups[4].append((test_data, test_label))

        # val_groups = [[], [], [], [], []]
        # for step in range(5):
        #     old_classes_propotion = step / (step + 1)
        #     new_classes_propotion = 1 / (step + 1)
        #     num_of_old_classes = int(2000 * old_classes_propotion)
        #     num_of_new_classes = int(2000 * new_classes_propotion)
        #
        #     if step >= 1:
        #         old_classes = []
        #         for i in range(step):
        #             old_classes.extend(test_groups[i])
        #         assert (len(old_classes) == 2000 * step)
        #         random.shuffle(old_classes)
        #         val_groups[step].extend(old_classes[:num_of_old_classes])
        #
        #     random.shuffle(test_groups[step])
        #     val_groups[step].extend(test_groups[step][:num_of_new_classes])
        #     assert (len(val_groups[step]) == 2000 or len(val_groups[step]) == 1999)
        return train_groups, test_groups  # , val_groups

    def getNextClasses(self, step_b):
        return self.train_groups[step_b], self.test_groups[step_b]


class CIFAR100_IncrementalDataset(datasets.Cifar100):
    def __init__(self,
                 root,
                 mode='train',
                 transform=None,
                 classes=range(20)):

        super(CIFAR100_IncrementalDataset, self).__init__(root,
                                                          mode,
                                                          transform,
                                                          download=True)
        self.mode = mode
        self.targets = []
        new_data = []
        for i in range(len(self.data)):
            image, label = self.data[i]
            image = np.reshape(image, [3, 32, 32])
            image = image.transpose([1, 2, 0])
            if self.backend == 'pil':
                image = Image.fromarray(image.astype('uint8'))
            if self.transform is not None:
                image = self.transform(image)
            if self.backend == 'pil':
                new_data.append(image)
                self.targets.append(np.array(label).astype('int64'))
            else:
                new_data.append(image.astype(self.dtype))
                self.targets.append(np.array(label).astype('int64'))

        self.data = new_data

        if mode == 'train':
            train_data = []
            train_labels = []
            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    train_data.append(self.data[i].astype(np.float32))
                    train_labels.append(self.targets[i])

            self.TrainData = train_data
            self.TrainLabels = train_labels

        else:
            test_data = []
            test_labels = []
            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    test_data.append(self.data[i].astype(np.float32))
                    test_labels.append(self.targets[i])

            self.TestData = test_data
            self.TestLabels = test_labels

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.TrainData[index], self.TrainLabels[index]
        else:
            return self.TestData[index], self.TestLabels[index]

    def __len__(self):
        if self.mode:
            return len(self.TrainLabels)
        else:
            return len(self.TestLabels)


def convert_binary(tensor):
    return tensor * (tensor == 1).long()


def convert_one_hot(target, num_classes):
    a = paddle.zeros([num_classes])
    a[target] = 1
    a = a.long()
    return a
    # return torch.LongTensor(a)


class CelebA(paddle.io.Dataset):
    def __init__(self, root, mode, transform=None, target_transform=convert_binary, loader=pil_loader):
        if self._checkIntegrity(root):
            print('Files already downloaded and verified.')
        else:
            self._extract(root, mode)
        images, targets = pickle.load(open('{}/{}.pkl'.format(root, mode), 'rb'))

        self.images = [os.path.join(root, 'Img/img_align_celeba', img) for img in images]
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def _extract(self, root, mode):
        images = []
        targets = []
        if mode == 'train':
            flag = '0'
        elif mode == 'valid':
            flag = '1'
        elif mode == 'test':
            flag = '2'
        else:
            raise RuntimeError("# Mode error")

        for line in open(os.path.join(root, './Eval/list_eval_partition.txt', ), 'r'):
            sample = line.split()
            if sample[1] == flag:
                images.append(sample[0])

        for idx, line in enumerate(open(os.path.join(root, './Anno/list_attr_celeba.txt', ), 'r')):
            sample = line.split()
            if idx <= 1:
                continue
            if len(sample) != 41:
                raise (RuntimeError("# Annotated face attributes of CelebA dataset should not be different from 40"))
            if sample[0] in images:
                targets.append([int(i) for i in sample[1:]])
        pickle.dump((images, targets), open('{}/{}.pkl'.format(root, mode), 'wb'))

    def _checkIntegrity(self, root):
        return (os.path.isfile('{}/train.pkl'.format(root))
                and os.path.isfile('{}/test.pkl'.format(root))
                and os.path.isfile('{}/valid.pkl'.format(root)))

    def __getitem__(self, index):
        path = self.images[index]
        sample = self.loader(path)
        target = self.targets[index]
        # target = torch.LongTensor(target)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.images)


import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive, check_integrity, download_url, verify_str_arg


class TinyImageNet(VisionDataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'tiny-imagenet-200/'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)

        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))

        if self._check_integrity():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        if not os.path.isdir(self.dataset_path):
            print('Extracting...')
            extract_archive(os.path.join(root, self.filename))

        _, class_to_idx = find_classes(os.path.join(self.dataset_path, 'wnids.txt'))

        self.data = make_dataset(self.root, self.base_folder, self.split, class_to_idx)

    def _download(self):
        print('Downloading...')
        download_url(self.url, root=self.root, filename=self.filename)
        print('Extracting...')
        extract_archive(os.path.join(self.root, self.filename))

    def _check_integrity(self):
        return check_integrity(os.path.join(self.root, self.filename), self.md5)

    def __getitem__(self, index):
        img_path, target = self.data[index]
        image = self.loader(img_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.data)


def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s: s.strip(), r.readlines()))

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


def make_dataset(root, base_folder, dirname, class_to_idx):
    images = []
    dir_path = os.path.join(root, base_folder, dirname)

    if dirname == 'train':
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                for imgname in sorted(os.listdir(cls_imgs_path)):
                    path = os.path.join(cls_imgs_path, imgname)
                    item = (path, class_to_idx[fname])
                    images.append(item)
    else:
        imgs_path = os.path.join(dir_path, 'images')
        imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}

        for imgname in sorted(os.listdir(imgs_path)):
            path = os.path.join(imgs_path, imgname)
            item = (path, class_to_idx[cls_map[imgname]])
            images.append(item)

    return images


if __name__ == '__main__':
    train_dataset = TinyImageNet('./datasets/tiny-imagenet', split='train', download=True)
    test_dataset = TinyImageNet('./datasets/tiny-imagenet', split='val', download=True)
