from __future__ import print_function
import json
import os
import paddle
import paddle.fluid as fluid
import numpy
import sys
import six
from paddle.fluid import layers
import zmq

def data_generater(data):
        def train_data():
                for i in range(len(data)):
                        conv = data[i][0]
                        label = data[i][1]
                        yield conv,label

#       def test_data():
#               test_file = open('cifar10_test.json','r')
#                data = json.load(test_file)
#                print(len(data))
#                for i in range(len(data)):
#                        conv = data[i][0]
#                        label = data[i][1]
#                        yield conv,label

        return train_data

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

def shortcut(input, ch_in, ch_out, stride):
    if ch_in != ch_out:
        return conv_bn_layer(input, ch_out, 1, stride, 0, None)
    else:
        return input

def basicblock(input, ch_in, ch_out, stride):
    tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
    short = shortcut(input, ch_in, ch_out, stride)
    return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')

def layer_warp(block_func, input, ch_in, ch_out, count, stride):
    tmp = block_func(input, ch_in, ch_out, stride)
    for i in range(1, count):
        tmp = block_func(tmp, ch_out, ch_out, 1)
    return tmp

def resnet_cifar10(ipt, depth=32):
    # depth should be one of 20, 32, 44, 56, 110, 1202
    assert (depth - 2) % 6 == 0
    n = (depth - 2) // 6
    nStages = {16, 64, 128}
    res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
    pool = fluid.layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    predict = fluid.layers.fc(input=pool, size=10, act='softmax')
    return predict

def train_test(program, reader):
    count = 0
    feed_var_list = [
        program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder_test = fluid.DataFeeder(
        feed_list=feed_var_list, place=place)
    test_exe = fluid.Executor(place)
    accumulated = len([avg_cost, acc]) * [0]
    for tid, test_data in enumerate(reader()):
        avg_cost_np = test_exe.run(program=program,
                                   feed=feeder_test.feed(test_data),
                                   fetch_list=[loss.name, accuracy.name])
        accumulated = [x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)]
        count += 1
    return [x / count for x in accumulated]

#train_file = open('test.json','r')
#train_file = open('t.json','r')
#data = json.load(train_file)

conv1 = fluid.layers.data(name='conv1', shape=[16,32,32], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
place = fluid.CUDAPlace(0)
predicts = resnet_cifar10(conv1)
feeder = fluid.DataFeeder(place=place, feed_list=[conv1, label])


cost = fluid.layers.cross_entropy(input=predicts, label=label)
accuracy = fluid.layers.accuracy(input=predicts, label=label)
loss = fluid.layers.mean(cost)
test_program = fluid.default_main_program().clone(for_test=True)


optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(loss)


exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

message = json.loads(socket.recv())
print(numpy.array(message[0][0]).shape)
socket.send("data received")
train_data = data_generater(message)
train_reader = paddle.batch(
    paddle.reader.shuffle(
        train_data, buf_size=5000),
    batch_size=64)

EPOCH_NUM = 200
step = 0
for pass_id in range(EPOCH_NUM):
    count = 0
    avg_loss = 0
    avg_acc = 0
    for step_id, data_train in enumerate(train_reader()):
        loss_value, acc_value = exe.run(fluid.default_main_program(),
                                 feed=feeder.feed(data_train),
                                 fetch_list=[loss.name, accuracy.name])
        count += 1
        step += 1
        if step % 10 == 0:
            print("epoch: "+ str(pass_id)+"step: "+str(step)+"loss: "+str(loss_value)+"acc: "+str(acc_value))

        avg_loss += loss_value
        avg_acc += acc_value
    print(avg_loss/count,avg_acc/count)

#    avg_cost_test, accuracy_test = train_test(test_program,
 #                                             reader=test_reader)

#    print(avg_cost_test, accuracy_test)
