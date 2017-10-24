import time

import numpy as np
import dynet as dy
# import tensorflow as tf

from q1_softmax import softmax
from q1_softmax import cross_entropy_loss
from model import Model
from utils.general_utils import get_minibatches


class Config(object):
    """Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_samples = 1024
    n_features = 100
    n_classes = 5
    batch_size = 64
    n_epochs = 50
    lr = 1e-4


class SoftmaxModel(object):
    def __init__(self, config, m):
        self.config = config
        self.pW = m.add_parameters((config.n_features, config.n_classes))
        self.pb = m.add_parameters((config.n_classes,))

    def create_network_return_loss(self, input, label):
        self.input = dy.inputTensor(input)
        self.label = dy.inputTensor(label)
        W = dy.parameter(self.pW)
        y = softmax(self.input * W + self.label)
        return cross_entropy_loss(self.label, y)


def test_softmax_model():
    """Train softmax model for a number of steps."""
    config = Config()

    # Generate random data to train the model on
    np.random.seed(1234)
    inputs = np.random.rand(config.n_samples, config.n_features)
    labels = np.zeros((config.n_samples, config.n_classes), dtype=np.int32)
    labels[:, 1] = 1
    # for i in xrange(config.n_samples):
    #    labels[i, i%config.n_classes] = 1

    mini_batches = [
        [inputs[k:k + config.batch_size], labels[k:k + config.batch_size]]
        for k in xrange(0, config.n_samples, config.batch_size)]

    m = dy.ParameterCollection()
    trainer = dy.SimpleSGDTrainer(m)
    trainer.learning_rate = config.lr
    net = SoftmaxModel(config, m)
    for epoch in range(config.n_epochs):
        start_time = time.time()
        for mini_batch in mini_batches:
            dy.renew_cg()
            losses = []
            for ix in xrange(config.batch_size):
                l = net.create_network_return_loss(
                    np.array(mini_batch[0][ix]).reshape(1, config.n_features),
                    np.array(mini_batch[1][ix]).reshape(1, config.n_classes))
                losses.append(l)
            loss = dy.esum(losses) / config.batch_size
            loss.forward()
            loss.backward()
            trainer.update()
        duration = time.time() - start_time
        print 'Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, loss.value(), duration)

    print loss.value()
    assert loss.value() < .5
    print "Basic (non-exhaustive) classifier tests pass"


if __name__ == "__main__":
    test_softmax_model()