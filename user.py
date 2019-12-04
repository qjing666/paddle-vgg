from __future__ import print_function
import os
import paddle
import paddle.fluid as fluid
import numpy
import sys
import six
from paddle.fluid import layers
import json
import zmq


def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act='relu',
                  bias_attr=False):
    tmp = fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=bias_attr)
    return fluid.layers.batch_norm(input=tmp, act=act)




images = fluid.layers.data(name='images', shape=[3,32,32], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
place = fluid.CPUPlace()
conv1 = conv_bn_layer(images, ch_out=16, filter_size=3, stride=1, padding=1)
feeder = fluid.DataFeeder(place=place, feed_list=[images,label])
pretrained_model = '../cifar_res'
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

def if_exist(var):
        return os.path.exists(os.path.join(pretrained_model, var.name))
fluid.io.load_vars(exe, pretrained_model, main_program=fluid.default_main_program(),
                  predicate=if_exist)

train_data = paddle.dataset.cifar.train10()
test_data = paddle.dataset.cifar.test10()
train_set = []

step = 0
for data in train_data():
    print("%dstart" % step)
    pre_data = []
    pre_data.append(data)
    conv = exe.run(program=fluid.default_main_program(),feed=feeder.feed(pre_data), fetch_list=[conv1.name])
    train_set.append([conv[0][0].tolist(),data[1]])
    step += 1
    if step == 2:
    	break
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")
socket.send(json.dumps(train_set))
response = socket.recv()
print(response)

#target_file = 't.json'
#with open(target_file,'w') as f:
#        json.dump(train_set,f)
