import paddle
import os
import paddle.fluid as fluid
import numpy
import sys


class VGGNet():
    def conv_block(self, input, num_filter, groups, name=None):
        conv = input
        for i in range(groups):
            conv = fluid.layers.conv2d(
                input=conv,
                num_filters=num_filter,
                filter_size=3,
                stride=1,
                padding=1,
                act='relu',
                param_attr=fluid.param_attr.ParamAttr(
                    name=name + str(i + 1) + "_weights"),
                bias_attr=False)
        return fluid.layers.pool2d(
            input=conv, pool_size=2, pool_type='max', pool_stride=2)

    def net(self, input, class_dim=10):
	#resize_input = fluid.layers.resize_bilinear(input, out_shape=[224, 224])
        conv1 = self.conv_block(input, 64, 2, name="conv1_")
        conv2 = self.conv_block(conv1, 128, 2, name="conv2_")
        conv3 = self.conv_block(conv2, 256, 3, name="conv3_")
        conv4 = self.conv_block(conv3, 512, 3, name="conv4_")
        conv5 = self.conv_block(conv4, 512, 3, name="conv5_")	
	
	fc_dim = 4096
        fc1 = fluid.layers.fc(
            input=conv5,
            size=fc_dim,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(name="fc6_weights"),
            bias_attr=fluid.param_attr.ParamAttr(name="fc6_offset"))
        fc1 = fluid.layers.dropout(x=fc1, dropout_prob=0.5)
        fc2 = fluid.layers.fc(
            input=fc1,
            size=fc_dim,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(name="fc7_weights"),
            bias_attr=fluid.param_attr.ParamAttr(name="fc7_offset"))
        fc2 = fluid.layers.dropout(x=fc2, dropout_prob=0.5)
        predicts = fluid.layers.fc(
            input=fc2,
            size=class_dim,
            param_attr=fluid.param_attr.ParamAttr(name="fc8_weights"),
            bias_attr=fluid.param_attr.ParamAttr(name="fc8_offset"))

        return predicts

images = fluid.layers.data(name='images', shape=[1,28,28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

vggnet = VGGNet()
predicts = vggnet.net(images,10)
cost = fluid.layers.cross_entropy(input=predicts, label=label)
accuracy = fluid.layers.accuracy(input=predicts, label=label)
loss = fluid.layers.mean(cost)
test_program = fluid.default_main_program().clone(for_test=True)


optimizer = fluid.optimizer.Adam(learning_rate=0.01)
optimizer.minimize(loss)
#place = fluid.CPUPlace()
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=5000),
    batch_size=64)

test_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.test(), buf_size=500),
    batch_size=64)

epoch = 100
step = 0
pretrained_model = '../test'
#place = fluid.CPUPlace()
place = fluid.CUDAPlace(0)
feeder = fluid.DataFeeder(place=place, feed_list=['images', 'label'])
exe.run(fluid.default_startup_program())


def if_exist(var):
    return os.path.exists(os.path.join(pretrained_model, var.name))

#fluid.io.load_vars(exe, pretrained_model, main_program=fluid.default_main_program(),
#                  predicate=if_exist)


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

	

