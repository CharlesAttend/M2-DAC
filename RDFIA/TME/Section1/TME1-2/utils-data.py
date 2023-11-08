import io
import imageio
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.io


class CirclesData:
    def __init__(self):
        # Grid
        x1, x2 = np.meshgrid(np.arange(-2, 2, 0.1), np.arange(-2, 2, 0.1))
        self._Xgrid = np.array([x1.flatten(), x2.flatten()]).T.astype("float32")

        # GIF creation
        self.gif_frames = []

        # Real data
        circles = scipy.io.loadmat("circles.mat")
        self._Xtrain, self._Xtest, self._Ytrain, self._Ytest = (
            circles["Xtrain"].astype("float32"),
            circles["Xtest"].astype("float32"),
            circles["Ytrain"].astype("float32"),
            circles["Ytest"].astype("float32"),
        )

        self._Xgrid_th = torch.from_numpy(self._Xgrid)
        self._Xtrain_th = torch.from_numpy(self._Xtrain)
        self._Xtest_th = torch.from_numpy(self._Xtest)
        self._Ytrain_th = torch.from_numpy(self._Ytrain)
        self._Ytest_th = torch.from_numpy(self._Ytest)

        self.loss_train = []
        self.loss_test = []
        self.acc_train = []
        self.acc_test = []

    def __getattr__(self, key):
        if key == "Xgrid":
            return self._Xgrid_th
        if key == "Xtrain":
            return self._Xtrain_th
        if key == "Xtest":
            return self._Xtest_th
        if key == "Ytrain":
            return self._Ytrain_th
        if key == "Ytest":
            return self._Ytest_th
        return None

    def plot_data(self):
        plt.figure(1, figsize=(5, 5))
        plt.plot(
            self._Xtrain[self._Ytrain[:, 0] == 1, 0],
            self._Xtrain[self._Ytrain[:, 0] == 1, 1],
            "bo",
            label="Train",
        )
        plt.plot(
            self._Xtrain[self._Ytrain[:, 1] == 1, 0],
            self._Xtrain[self._Ytrain[:, 1] == 1, 1],
            "ro",
        )
        plt.plot(
            self._Xtest[self._Ytest[:, 0] == 1, 0],
            self._Xtest[self._Ytest[:, 0] == 1, 1],
            "b+",
            label="Test",
        )
        plt.plot(
            self._Xtest[self._Ytest[:, 1] == 1, 0],
            self._Xtest[self._Ytest[:, 1] == 1, 1],
            "r+",
        )
        plt.legend()
        plt.show()

    def plot_data_with_grid(self, Ygrid, title=""):
        plt.figure(2, figsize=(5, 5))
        Ygrid = Ygrid[:, 1].numpy()
        plt.clf()
        plt.imshow(np.reshape(Ygrid, (40, 40)))
        plt.plot(
            self._Xtrain[self._Ytrain[:, 0] == 1, 0] * 10 + 20,
            self._Xtrain[self._Ytrain[:, 0] == 1, 1] * 10 + 20,
            "bo",
            label="Train",
        )
        plt.plot(
            self._Xtrain[self._Ytrain[:, 1] == 1, 0] * 10 + 20,
            self._Xtrain[self._Ytrain[:, 1] == 1, 1] * 10 + 20,
            "ro",
        )
        plt.plot(
            self._Xtest[self._Ytest[:, 0] == 1, 0] * 10 + 20,
            self._Xtest[self._Ytest[:, 0] == 1, 1] * 10 + 20,
            "b+",
            label="Test",
        )
        plt.plot(
            self._Xtest[self._Ytest[:, 1] == 1, 0] * 10 + 20,
            self._Xtest[self._Ytest[:, 1] == 1, 1] * 10 + 20,
            "r+",
        )
        plt.xlim(0, 39)
        plt.ylim(0, 39)
        plt.clim(0.3, 0.7)
        plt.title(title)
        plt.draw()
        # plt.pause(1e-3)
        # for GIF creation
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        self.gif_frames.append(imageio.imread(buf))
        plt.close()

    def plot_loss(self, loss_train, loss_test, acc_train, acc_test):
        self.loss_train.append(loss_train)
        self.loss_test.append(loss_test)
        self.acc_train.append(acc_train)
        self.acc_test.append(acc_test)
        plt.figure(3)
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.plot(np.array(self.acc_train), label="acc. train")
        plt.plot(np.array(self.acc_test), label="acc. test")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(np.array(self.loss_train), label="loss train")
        plt.plot(np.array(self.loss_test), label="loss test")
        plt.legend()
        plt.show()

    def save_gif(self, filename):
        imageio.mimsave(filename, self.gif_frames, duration=0.1, loop=0)


class MNISTData:
    def __init__(self):
        # Real data
        mnist = scipy.io.loadmat("mnist.mat")
        self._Xtrain_th = torch.from_numpy(mnist["Xtrain"].astype("float32"))
        self._Xtest_th = torch.from_numpy(mnist["Xtest"].astype("float32"))
        self._Ytrain_th = torch.from_numpy(mnist["Ytrain"].astype("float32"))
        self._Ytest_th = torch.from_numpy(mnist["Ytest"].astype("float32"))

    def __getattr__(self, key):
        if key == "Xtrain":
            return self._Xtrain_th
        if key == "Xtest":
            return self._Xtest_th
        if key == "Ytrain":
            return self._Ytrain_th
        if key == "Ytest":
            return self._Ytest_th
        return None
