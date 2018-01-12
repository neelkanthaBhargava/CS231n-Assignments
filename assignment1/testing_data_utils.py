import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

def main():
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    x_train, y_train, x_test, y_test = load_CIFAR10(cifar10_dir)


    print('Training data shape: ', x_train.shape)
    print('Training labels shape: ',y_train.shape)
    print('Test data shape: ',x_test.shape)
    print('Test labels shape: ',y_test.shape)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(x_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()


if __name__ == '__main__':
    main()
