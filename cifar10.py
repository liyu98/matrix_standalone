import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Take a look at some samples from the dataset: each class shows some
def showPic(X_train, y_train):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        # Randomly pick some from a category
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    showPic(x_train, y_train)
