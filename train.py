import datetime
import json
import os

import mxnet as mx
import numpy as np
from mxnet import gluon as gl
from mxnet import nd
from mxnet.gluon import nn

from data_utils import DogDataSet
from tensorboardX import SummaryWriter

im_path = './data/train/'


def transform_train(img):
    '''
    img is the mx.image.imread object
    '''
    img = img.astype('float32') / 255
    random_shape = int(np.random.uniform() * 224 + 256)
    # random samplely in [256, 480]
    aug_list = mx.image.CreateAugmenter(
        data_shape=(3, 224, 224),
        resize=random_shape,
        rand_mirror=True,
        rand_crop=True,
        mean=np.array([0.4736, 0.4504, 0.3909]),
        std=np.array([0.2655, 0.2607, 0.2650]))

    for aug in aug_list:
        img = aug(img)
    img = nd.transpose(img, (2, 0, 1))
    return img


def transform_valid(img):
    img = img.astype('float32') / 255.
    aug_list = mx.image.CreateAugmenter(
        data_shape=(3, 224, 224),
        mean=np.array([0.4736, 0.4504, 0.3909]),
        std=np.array([0.2655, 0.2607, 0.2650]))

    for aug in aug_list:
        img = aug(img)
    img = nd.transpose(img, (2, 0, 1))
    return img


# ## use DataLoader

train_json = './data/train.json'
train_set = DogDataSet(train_json, im_path, transform_train)
train_data = gl.data.DataLoader(
    train_set, batch_size=64, shuffle=True, last_batch='keep')

valid_json = './data/valid.json'
valid_set = DogDataSet(valid_json, im_path, transform_valid)
valid_data = gl.data.DataLoader(
    valid_set, batch_size=128, shuffle=False, last_batch='keep')

criterion = gl.loss.SoftmaxCrossEntropyLoss()

# ctx = [mx.gpu(0), mx.gpu(1)]
ctx = mx.gpu(0)
num_epochs = 250
lr = 0.1
wd = 1e-4
lr_period = 70
lr_decay = 0.1

net = gl.model_zoo.vision.resnet50_v2(pretrained=True, ctx=ctx)
for _, i in net.features.collect_params().items():
    i.lr_mult = 0.01
# net.load_params('./resnet_50.params', ctx=ctx)
new_classifier = nn.HybridSequential()
with new_classifier.name_scope():
    new_classifier.add(nn.BatchNorm(),
                       nn.Activation('relu'),
                       nn.GlobalAvgPool2D(), nn.Flatten(), nn.Dense(120))
new_classifier.initialize(init=mx.init.Xavier(), ctx=ctx)
net.classifier = new_classifier
# freeze weight
# for _, i in net.features.collect_params().items():
# i.grad_req = 'null'
# net.initialize(init=mx.init.Xavier(), ctx=ctx)
# net.collect_params().load('finetune_resnet_20.params', ctx=ctx)
net.hybridize()

writer = SummaryWriter()


def get_acc(output, label):
    pred = output.argmax(1)
    correct = (pred == label).sum()
    return correct.asscalar()


def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period,
          lr_decay):
    trainer = gl.Trainer(net.collect_params(), 'sgd',
                         {'learning_rate': lr,
                          'momentum': 0.9,
                          'wd': wd})

    prev_time = datetime.datetime.now()
    for epoch in range(20, num_epochs):
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        train_loss = 0
        correct = 0
        total = 0
        for data, label in train_data:
            bs = data.shape[0]
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with mx.autograd.record():
                # with mx.autograd.pause(train_mode=True):
                # data_feature = net.features(data)
                # output = net.classifier(data_feature)
                output = net(data)
                loss = criterion(output, label)
            loss.backward()
            trainer.step(bs)
            train_loss += loss.sum().asscalar()
            correct += get_acc(output, label)
            total += bs
        writer.add_scalars('loss', {'train': train_loss / total}, epoch)
        writer.add_scalars('acc', {'train': correct / total}, epoch)
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_correct = 0
            valid_total = 0
            valid_loss = 0
            for data, label in valid_data:
                bs = data.shape[0]
                data = data.as_in_context(ctx)
                label = label.as_in_context(ctx)
                output = net(data)
                loss = criterion(output, label)
                valid_loss += nd.sum(loss).asscalar()
                valid_correct += get_acc(output, label)
                valid_total += bs
            valid_acc = valid_correct / valid_total
            writer.add_scalars('loss', {'valid': valid_loss / valid_total},
                               epoch)
            writer.add_scalars('acc', {'valid': valid_acc}, epoch)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train acc %f, Valid Loss: %f, Valid acc %f, "
                % (epoch, train_loss / total, correct / total,
                   valid_loss / valid_total, valid_acc))
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, " %
                         (epoch, train_loss / total, correct / total))
        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))
        if (epoch + 1) % 10 == 0:
            net.collect_params().save(
                './finetune_resnet_{}.params'.format(epoch + 1))


train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period,
      lr_decay)

net.collect_params().save('./finetune_resnet.params')
