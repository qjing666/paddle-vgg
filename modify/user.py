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

def conv_block(ipt, num_filter, groups, dropouts,name=None):
    tmp = ipt
    conv_num_filter=[num_filter] * groups
    conv_batchnorm_drop_rate=dropouts
    conv_with_batchnorm=True
    conv_filter_size=3
    conv_act='relu'
    conv_with_batchnorm=True
    conv_padding=1
    def __extend_list__(obj):
        if not hasattr(obj, '__len__'):
            return [obj] * len(conv_num_filter)
        else:
            assert len(obj) == len(conv_num_filter)
            return obj
    conv_padding = __extend_list__(conv_padding)
    conv_filter_size = __extend_list__(conv_filter_size)
    conv_with_batchnorm = __extend_list__(conv_with_batchnorm)
    conv_batchnorm_drop_rate = __extend_list__(conv_batchnorm_drop_rate)
    for i in six.moves.range(len(conv_num_filter)):
        local_conv_act = conv_act
        if conv_with_batchnorm[i]:
            local_conv_act = None

        tmp = layers.conv2d(
            input=tmp,
            num_filters=conv_num_filter[i],
            filter_size=conv_filter_size[i],
            padding=conv_padding[i],
            param_attr=fluid.param_attr.ParamAttr(name=name + str(i + 1) + "_weights"),
            bias_attr=fluid.param_attr.ParamAttr(
                name=name + str(i + 1) + "_offset"),
            act=local_conv_act,
            use_cudnn=False)
        if conv_with_batchnorm[i]:
            tmp = layers.batch_norm(input=tmp, act=conv_act, in_place=True)
            drop_rate = conv_batchnorm_drop_rate[i]
            if abs(drop_rate) > 1e-5:
                tmp = layers.dropout(x=tmp, dropout_prob=drop_rate)
    pool_out = layers.pool2d(
         input=tmp,
         pool_size=2,
         pool_type='max',
         pool_stride=2,
         use_cudnn=False)
    return pool_out


def user_model(input):
    conv1 = conv_block(input, 64, 2, [0.3, 0],name="conv1_")
    return conv1




images = fluid.layers.data(name='images', shape=[1,28,28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
place = fluid.CPUPlace()
conv1 = user_model(images)
feeder = fluid.DataFeeder(place=place, feed_list=[images,label])
pretrained_model = '../test'
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())


def if_exist(var):
        return os.path.exists(os.path.join(pretrained_model, var.name))
fluid.io.load_vars(exe, pretrained_model, main_program=fluid.default_main_program(),
                  predicate=if_exist)


train_data = paddle.dataset.mnist.train()
test_data = paddle.dataset.mnist.test()
train_set = []
step = 0
for data in train_data():
    print("%dstart" % step)
    pre_data = []
    pre_data.append(data)
    conv = exe.run(program=fluid.default_main_program(),feed=feeder.feed(pre_data), fetch_list=[conv1.name])
    train_set.append([conv[0][0].tolist(),data[1]])
    step += 1

print(len(train_set),len(train_set[0]),len(train_set[0][0]),train_set[0][1])
#context = zmq.Context()
#socket = context.socket(zmq.REQ)
#socket.connect("tcp://localhost:5555")
#socket.send(json.dumps(train_set))
#response = socket.recv()
#print(response)
target_file = 'mnist_train.json'
with open(target_file,'w') as f:
	json.dump(train_set,f)

