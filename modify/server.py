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
		train_file = open('cifar10_train.json','r')
        	data = json.load(train_file)
		for i in range(len(data)):
			conv = data[i][0]
			label = data[i][1]
			yield conv,label

#	def test_data():
#		test_file = open('cifar10_test.json','r')
#                data = json.load(test_file)
#                print(len(data))
#                for i in range(len(data)):
#                        conv = data[i][0]
#                        label = data[i][1]
#                        yield conv,label

	return train_data



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
            use_cudnn=True)
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
         use_cudnn=True)
    return pool_out



def server_model(conv1):
#    resize_input = fluid.layers.resize_bilinear(input, out_shape=[224, 224])
#    conv1 = conv_block(resize_input, 64, 2, [0.3, 0],name="conv1_")
    conv2 = conv_block(conv1, 128, 2, [0.4, 0],name="conv2_")
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0],name="conv3_")
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0],name="conv4_")
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0],name="conv5_")

    fc_dim = 4096
    fc_name = ["fc6", "fc7", "fc8"]

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(
                       input=drop,
                       size=fc_dim,
                       act=None,
                       param_attr=fluid.param_attr.ParamAttr(name=fc_name[0] + "_weights"),
                       bias_attr=fluid.param_attr.ParamAttr(name=fc_name[0] + "_offset"))
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(
                       input=drop2,
                       size=fc_dim,
                       act=None,
                       param_attr=fluid.param_attr.ParamAttr(name=fc_name[1] + "_weights"),
                       bias_attr=fluid.param_attr.ParamAttr(name=fc_name[1] + "_offset"))
    predict = fluid.layers.fc(
                           input=fc2,
                           size=10,
                           act='softmax',
                           param_attr=fluid.param_attr.ParamAttr(name=fc_name[2] + "_weights"),
                           bias_attr=fluid.param_attr.ParamAttr(name=fc_name[2] + "_offset"))
    return predict



#context = zmq.Context()
#socket = context.socket(zmq.REP)
#socket.bind("tcp://*:5555")

#message = json.loads(socket.recv())
#print(numpy.array(message[0][0]).shape)
#socket.send("data received")



conv1 = fluid.layers.data(name='conv1', shape=[64,112,112], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
place = fluid.CUDAPlace(0)
predicts = server_model(conv1)
feeder = fluid.DataFeeder(place=place, feed_list=[conv1, label])


cost = fluid.layers.cross_entropy(input=predicts, label=label)
accuracy = fluid.layers.accuracy(input=predicts, label=label)
loss = fluid.layers.mean(cost)
test_program = fluid.default_main_program().clone(for_test=True)
epoch = 100
step = 0
optimizer = fluid.optimizer.SGD(learning_rate=0.01)
optimizer.minimize(loss)

pretrained_model = '../test'
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

train_data = data_generater(message)

train_reader = paddle.batch(
    paddle.reader.shuffle(
        train_data, buf_size=5000),
    batch_size=64)

def if_exist(var):
    return os.path.exists(os.path.join(pretrained_model, var.name))

fluid.io.load_vars(exe, pretrained_model, main_program=fluid.default_main_program(),
                  predicate=if_exist)



for i in range(epoch):
    print("training epoch %d"%i)
    for data in train_reader():
        avg_loss, acc = exe.run(program=fluid.default_main_program(),feed=feeder.feed(data), fetch_list=[loss.name,accuracy.name])
        step += 1
        if step % 1 == 0:
            print("epoch: "+ str(i)+"step: "+str(step)+"loss: "+str(avg_loss[0])+"acc: "+str(acc))

print("program end")
