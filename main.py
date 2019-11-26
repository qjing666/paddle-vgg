from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy
import sys

def vgg_bn_drop(input):
    def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=ipt,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    resize_input = fluid.layers.resize_bilinear(input, out_shape=[224, 224])
    conv1 = conv_block(resize_input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    predict = fluid.layers.fc(input=fc2, size=10, act='softmax')
    return predict



images = fluid.layers.data(name='images', shape=[3,32,32], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
place = fluid.CUDAPlace(0)
feeder = fluid.DataFeeder(place=place, feed_list=['images', 'label'])
predicts = vgg_bn_drop(images)

cost = fluid.layers.cross_entropy(input=predicts, label=label)
accuracy = fluid.layers.accuracy(input=predicts, label=label)
loss = fluid.layers.mean(cost)
test_program = fluid.default_main_program().clone(for_test=True)
epoch = 100
step = 0
optimizer = fluid.optimizer.SGD(learning_rate=0.000001)
optimizer.minimize(loss)
#place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())


train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.cifar.train10(), buf_size=5000),
    batch_size=64)

test_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.cifar.test10(), buf_size=500),
    batch_size=64)


for i in range(epoch):
    print("training epoch %d"%i)
    for data in train_reader():
        avg_loss, acc = exe.run(program=fluid.default_main_program(),feed=feeder.feed(data), fetch_list=[loss.name,accuracy.name])
        step += 1
        if step % 10 == 0:
                print("epoch: "+ str(i)+"step: "+str(step)+"loss: "+str(avg_loss[0])+"acc: "+str(acc))
        #print("epoch: %d; step: %d;loss: %d;acc: ;" % (i,step,avg_loss[0],i))


    acc_set = []
    for test_data in test_reader():
        acc_np = exe.run(program=test_program,
                        feed=feeder.feed(test_data),
                        fetch_list=["accuracy_0.tmp_0"])
        acc_set.append(float(acc_np[0]))
    acc_val = numpy.array(acc_set).mean()
    print("Test with epoch %d, accuracy: %s" % (i, acc_val))



