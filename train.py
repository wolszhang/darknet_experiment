import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
import numpy as np
import os
import time
import networkx as nx

def graph_generator(model, graph_param, save_path, file_name):
    graph_param[0] = int(graph_param[0])
    if model == 'ws':
        graph_param[1] = int(graph_param[1])
        graph = nx.random_graphs.connected_watts_strogatz_graph(*graph_param)
    elif model == 'er':
        graph = nx.random_graphs.erdos_renyi_graph(*graph_param)
    elif model == 'ba':
        graph_param[1] = int(graph_param[1])
        graph = nx.random_graphs.barabasi_albert_graph(*graph_param)

    if os.path.isfile(save_path + '/' + file_name + '.yaml') is True:
        print('graph loaded')
        dgraph = nx.read_yaml(save_path + '/' + file_name + '.yaml')

    else:
        dgraph = nx.DiGraph()
        dgraph.add_nodes_from(graph.nodes)
        dgraph.add_edges_from(graph.edges)

    in_node = []
    out_node = []
    for indeg, outdeg in zip(dgraph.in_degree, dgraph.out_degree):
        if indeg[1] == 0:
            in_node.append(indeg[0])
        elif outdeg[1] == 0:
            out_node.append(outdeg[0])
    # print(in_node, out_node)
    sorted = list(nx.topological_sort(dgraph))
    # nx.draw(dgraph)
    # plt.draw()
    # plt.show()

    if os.path.isdir(save_path) is False:
        os.makedirs(save_path)

    if os.path.isfile(save_path + '/' + file_name + '.yaml') is False:
        print('graph_saved')
        nx.write_yaml(dgraph, save_path + '/' + file_name + '.yaml')

    return dgraph, sorted, in_node, out_node



def parse_tfrecord(record):
    keys_to_features = {
        'height': tf.FixedLenFeature((), tf.int64),
        'width': tf.FixedLenFeature((), tf.int64),
        'image/raw': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64),
    }

    features = tf.parse_single_example(record, features=keys_to_features)
    height = tf.cast(features['height'], tf.int64)
    width = tf.cast(features['width'], tf.int64)
    image = tf.cast(features['image/raw'], tf.string)
    label = tf.cast(features['label'], tf.int64)

    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, shape=[height, width, -1])

    return image, label

def flip(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label

def pad_and_crop(image, label, shape, pad_size=2):
    image = tf.image.pad_to_bounding_box(image, pad_size, pad_size, shape[0]+pad_size*2, shape[1]+pad_size*2)
    image = tf.image.random_crop(image, shape)
    return image, label

def standardization(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label

# create batch iterator
def batch_iterator(dataset_dir, epochs, batch_size, augmentation=None, training=False, drop_remainder=True):
    if os.path.isfile(dataset_dir) is False:
        raise FileNotFoundError(dataset_dir, 'not exist')
    dataset = tf.data.TFRecordDataset(dataset_dir)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.map(standardization)
    if training is True:
        dataset = dataset.shuffle(100000)
        dataset = dataset.repeat(epochs)
        if augmentation is not None:
            for aug_func in augmentation:
                dataset = dataset.map(aug_func, num_parallel_calls=len(augmentation))
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    iterator = dataset.make_initializable_iterator()

    return iterator

def conv_block(input, kernels, filters, strides, dropout_rate, training, scope):
    with tf.variable_scope(scope):
        input = tf.nn.relu(input)
        input = tf.layers.separable_conv2d(input, filters=filters, kernel_size=[kernels, kernels], strides=[strides, strides], padding='SAME')
        input = tf.layers.batch_normalization(input, training=training)
        input = tf.layers.dropout(input, rate=dropout_rate, training=training)
    return input

def build_stage(input, filters, dropout_rate, training, graph_data, scope):
    graph, graph_order, start_node, end_node = graph_data

    interms = {}
    with tf.variable_scope(scope):
        for node in graph_order:
            if node in start_node:
                interm = conv_block(input, 3, filters, 2, dropout_rate, training, scope='node' + str(node))
                interms[node] = interm
            else:
                in_node = list(nx.ancestors(graph, node))
                if len(in_node) > 1:
                    with tf.variable_scope('node' + str(node)):
                        weight = tf.get_variable('sum_weight', shape=len(in_node), dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
                        weight = tf.nn.sigmoid(weight)
                        interm = weight[0] * interms[in_node[0]]
                        for idx in range(1, len(in_node)):
                            interm += weight[idx] * interms[in_node[idx]]
                        interm = conv_block(interm, 3, filters, 1, dropout_rate, training, scope='conv_block' + str(node))
                        interms[node] = interm
                elif len(in_node) == 1:
                    interm = conv_block(interms[in_node[0]], 3, filters, 1, dropout_rate, training, scope='node' + str(node))
                    interms[node] = interm

        output = interms[end_node[0]]
        for idx in range(1, len(end_node)):
            output += interms[end_node[idx]]

        return output

def small_regime(input, stages, filters, classes, dropout_rate, graph_model, graph_param, graph_file_path, init_subsample, training):
    with tf.variable_scope('conv1'):
        if init_subsample is True:
            input = tf.layers.separable_conv2d(input, filters=int(filters/2), kernel_size=[3, 3], strides=[2, 2], padding='SAME')
            input = tf.layers.batch_normalization(input, training=training)
        else:
            input = tf.layers.separable_conv2d(input, filters=int(filters / 2), kernel_size=[3, 3], strides=[1, 1],
                                               padding='SAME')

    input = conv_block(input, 3, filters, 2, dropout_rate, training, 'conv2')

    for stage in range(3, stages+1):
        graph_data = graph_generator(graph_model, graph_param, graph_file_path, 'conv' + str(stage) + '_' + graph_model)
        input = build_stage(input, filters, dropout_rate, training, graph_data, 'conv' + str(stage))
        filters *= 2

    with tf.variable_scope('classifier'):
        input = conv_block(input, 1, 1280, 1, dropout_rate, training, 'conv_block_classifier')
        input = tf.layers.average_pooling2d(input, pool_size=input.shape[1:3], strides=[1, 1])
        input = tf.layers.flatten(input)
        input = tf.layers.dense(input, units=classes)

    return input

def regular_regime(input, stages, filters, classes, dropout_rate, graph_model, graph_param, graph_file_path, training):
    with tf.variable_scope('conv1'):
        input = tf.layers.separable_conv2d(input, filters=int(filters/2), kernel_size=[3, 3], strides=[2, 2], padding='SAME')
        input = tf.layers.batch_normalization(input, training=training)

    for stage in range(2, stages+1):
        graph_data = graph_generator(graph_model, graph_param, graph_file_path, 'conv' + str(stage) + '_' + graph_model)
        input = build_stage(input, filters, dropout_rate, training, graph_data, 'conv' + str(stage))
        filters *= 2
    
    with tf.variable_scope('classifier'):
        input = conv_block(input, 1, 1280, 1, dropout_rate, training, 'conv_block_classifier')
        input = tf.layers.average_pooling2d(input, pool_size=input.shape[1:3], strides=[1, 1])
        input = tf.layers.flatten(input)
        input = tf.layers.dense(input, units=classes)
        input = tf.layers.dropout(input, rate=dropout_rate)

    return input

def my_regime(input, stages, filters, classes, dropout_rate, graph_model, graph_param, graph_file_path, init_subsample, training): #regular regime 기반
    with tf.variable_scope('conv1'):
        if init_subsample is True:
            input = tf.layers.separable_conv2d(input, filters=int(filters/2), kernel_size=[3, 3], strides=[2, 2], padding='SAME')
            input = tf.layers.batch_normalization(input, training=training)
        else:
            input = tf.layers.separable_conv2d(input, filters=int(filters / 2), kernel_size=[3, 3], strides=[1, 1],
                                               padding='SAME')
            input = tf.layers.batch_normalization(input, training=training)

    for stage in range(2, stages+1):
        graph_data = graph_generator(graph_model, graph_param, graph_file_path, 'conv' + str(stage) + '_' + graph_model)
        input = build_stage(input, filters, dropout_rate, training, graph_data, 'conv' + str(stage))
        filters *= 2

    with tf.variable_scope('classifier'):
        input = conv_block(input, 1, 1280, 1, dropout_rate, training, 'conv_block_classifier')
        input = tf.layers.average_pooling2d(input, pool_size=input.shape[1:3], strides=[1, 1])
        input = tf.layers.flatten(input)
        input = tf.layers.dropout(input, rate=0.3, training=training)
        input = tf.layers.dense(input, units=classes)

    return input

def my_small_regime(input, stages, filters, classes, dropout_rate, graph_model, graph_param, graph_file_path, init_subsample, training): #regular regime 기반
    with tf.variable_scope('conv1'):
        if init_subsample is True:
            input = tf.layers.separable_conv2d(input, filters=int(filters/2), kernel_size=[3, 3], strides=[2, 2], padding='SAME')
            input = tf.layers.batch_normalization(input, training=training)
        else:
            input = tf.layers.separable_conv2d(input, filters=int(filters / 2), kernel_size=[3, 3], strides=[1, 1],
                                               padding='SAME')
            input = tf.layers.batch_normalization(input, training=training)

    input = conv_block(input, 3, filters, 1, dropout_rate, training, 'conv2')

    for stage in range(3, stages+1):
        graph_data = graph_generator(graph_model, graph_param, graph_file_path, 'conv' + str(stage) + '_' + graph_model)
        input = build_stage(input, filters, dropout_rate, training, graph_data, 'conv' + str(stage))
        filters *= 2

    with tf.variable_scope('classifier'):
        input = conv_block(input, 1, 1280, 1, dropout_rate, training, 'conv_block_classifier')
        input = tf.layers.average_pooling2d(input, pool_size=input.shape[1:3], strides=[1, 1])
        input = tf.layers.flatten(input)
        input = tf.layers.dropout(input, rate=0.3, training=training)
        input = tf.layers.dense(input, units=classes)

    return input

# argument parser for options
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_num', type=int, default=10, help='number of class')  # number of class
    parser.add_argument('--image_shape', type=int, nargs='+', default=[32, 32, 3], help='shape of image - height, width, channel')  # shape of image - height, width, channel
    parser.add_argument('--channel_count', type=int, default=78)
    parser.add_argument('--graph_model', type=str, default='ws')
    parser.add_argument('--graph_param', type=float, nargs='+', default=[32, 4, 0.75])
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate for dropout')  # dropout rate for dropout
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='initial learning rate')  # initial learning rate
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for momentum optimizer')  # momentum for momentum optimizer
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay factor')  # weight decay factor
    parser.add_argument('--train_set_size', type=int, default=50000, help='number of images for training set')  # number of images for training set
    parser.add_argument('--val_set_size', type=int, default=10000, help='number of images for validation set, 0 for skip validation')  # number of images for validation set, 0 for skip validation
    parser.add_argument('--batch_size', type=int, default=100, help='number of images for each batch')  # number of images for each batch
    parser.add_argument('--epochs', type=int, default=100, help='total epochs to train')  # total epochs to train
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='directory for checkpoint')  # directory for checkpoint
    parser.add_argument('--checkpoint_name', type=str, default='randwire_cifar10', help='filename for checkpoint')
    parser.add_argument('--train_record_dir', type=str, default='./dataset/cifar10/train.tfrecord', help='directory for training records')  # directory for training images
    parser.add_argument('--val_record_dir', type=str, default='./dataset/cifar10/test.tfrecord', help='directory for validation records')  # directory for training labels

    args = parser.parse_args()

    return args

# main function for training
def main(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('[Input Arguments]')
    for arg in args.__dict__:
        print(arg, '->', args.__dict__[arg])
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')

    st = time.time()
    images = tf.placeholder('float32', shape=[None, *args.image_shape], name='images')  # placeholder for images
    labels = tf.placeholder('float32', shape=[None, args.class_num], name='labels')  # placeholder for labels
    training = tf.placeholder('bool', name='training')  # placeholder for training boolean (is training)
    global_step = tf.get_variable(name='global_step', shape=[], dtype='int64', trainable=False)  # variable for global step
    best_accuracy = tf.get_variable(name='best_accuracy', dtype='float32', trainable=False, initializer=0.0)
    
    steps_per_epoch = round(args.train_set_size / args.batch_size)
    learning_rate = tf.train.piecewise_constant(global_step, [round(steps_per_epoch * 0.5 * args.epochs),
                                                              round(steps_per_epoch * 0.75 * args.epochs)],
                                                [args.learning_rate, 0.1 * args.learning_rate,
                                                 0.01 * args.learning_rate])
    # output logit from NN
    output = my_small_regime(images, 4, args.channel_count, args.class_num, args.dropout_rate,
                                args.graph_model, args.graph_param, args.checkpoint_dir + '/' + 'graphs', False, training)
    # output = RandWire.my_regime(images, 4, args.channel_count, args.class_num, args.dropout_rate,
    #                             args.graph_model, args.graph_param, args.checkpoint_dir + '/' + 'graphs', False, training)
    # output = RandWire.small_regime(images, 4, args.channel_count, args.class_num, args.dropout_rate,
    #                             args.graph_model, args.graph_param, args.checkpoint_dir + '/' + 'graphs', False,
    #                             training)
    # output = RandWire.regular_regime(images, 4, args.channel_count, args.class_num, args.dropout_rate,
    #                             args.graph_model, args.graph_param, args.checkpoint_dir + '/' + 'graphs', training)

    #loss and optimizer
    with tf.variable_scope('losses'):
        # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=output)
        loss = tf.losses.softmax_cross_entropy(labels, output, label_smoothing=0.1)
        loss = tf.reduce_mean(loss, name='loss')
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='l2_loss')

    with tf.variable_scope('optimizers'):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=args.momentum, use_nesterov=True)
        #optimizer = tf.train.AdamOptimizer(learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(loss + l2_loss * args.weight_decay, global_step=global_step)
        train_op = tf.group([train_op, update_ops], name='train_op')

    #accuracy
    with tf.variable_scope('accuracy'):
        output = tf.nn.softmax(output, name='output')
        prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1), name='prediction')
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='accuracy')

    # summary
    train_loss_summary = tf.summary.scalar("train_loss", loss)
    val_loss_summary = tf.summary.scalar("val_loss", loss)
    train_accuracy_summary = tf.summary.scalar("train_acc", accuracy)
    val_accuracy_summary = tf.summary.scalar("val_acc", accuracy)

    saver = tf.train.Saver()
    best_saver = tf.train.Saver()
    
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(args.checkpoint_dir + '/log', sess.graph)

        sess.run(tf.global_variables_initializer())
        augmentations = [lambda image, label: pad_and_crop(image, label, args.image_shape, 4), flip]
        train_iterator = batch_iterator(args.train_record_dir, args.epochs, args.batch_size, augmentations, True)
        train_images_batch, train_labels_batch = train_iterator.get_next()
        val_iterator = batch_iterator(args.val_record_dir, args.epochs, args.batch_size)
        val_images_batch, val_labels_batch = val_iterator.get_next()
        sess.run(train_iterator.initializer)
        if args.val_set_size != 0:
            sess.run(val_iterator.initializer)

        # restoring checkpoint
        try:
            saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
            print('checkpoint restored. train from checkpoint')
        except:
            print('failed to load checkpoint. train from the beginning')

        #get initial step
        gstep = sess.run(global_step)
        init_epoch = round(gstep / steps_per_epoch)
        init_epoch = int(init_epoch)

        for epoch_ in range(init_epoch + 1, args.epochs + 1):
            train_acc = []
            train_loss = []
            val_acc = []
            val_loss = []
            # train
            while gstep * args.batch_size < epoch_ * args.train_set_size:
                try:
                    train_images, train_labels = sess.run([train_images_batch, train_labels_batch])
                    train_labels = np.eye(args.class_num)[train_labels]
                    gstep, _, loss_, accuracy_, train_loss_sum, train_acc_sum = sess.run(
                        [global_step, train_op, loss, accuracy, train_loss_summary, train_accuracy_summary],
                        feed_dict={images: train_images, labels: train_labels, training: True})
                    train_loss.append(loss_)
                    train_acc.append(accuracy_)
                    writer.add_summary(train_loss_sum, gstep)
                    writer.add_summary(train_acc_sum, gstep)
                except tf.errors.OutOfRangeError:
                    break

            predictions = []

            # validation
            if args.val_set_size != 0:
                while True:
                    try:
                        val_images, val_labels = sess.run([val_images_batch, val_labels_batch])
                        val_labels = np.eye(args.class_num)[val_labels]
                        loss_, accuracy_, prediction_, val_loss_sum, val_acc_sum = sess.run(
                            [loss, accuracy, prediction, val_loss_summary, val_accuracy_summary],
                            feed_dict={images: val_images, labels: val_labels, training: False})
                        predictions.append(prediction_)
                        val_loss.append(loss_)
                        val_acc.append(accuracy_)
                        writer.add_summary(val_loss_sum, gstep)
                        writer.add_summary(val_acc_sum, gstep)
                    except tf.errors.OutOfRangeError:
                        sess.run(val_iterator.initializer)
                        break
            print('[ epoch %d/%d ]-> train acc: %.4f loss: %.4f val acc: %.4f loss: %.4f time: %.f'% (epoch_, args.epochs, np.average(train_acc), np.sum(train_loss), np.average(val_acc), np.sum(val_loss), time.time()-st))
            st = time.time()
            saver.save(sess, args.checkpoint_dir + '/' + args.checkpoint_name, global_step=global_step)

            predictions = np.concatenate(predictions)
            print('best: ', best_accuracy.eval(), '\ncurrent: ', np.mean(predictions))
            if best_accuracy.eval() < np.mean(predictions):
                print('save checkpoint')
                best_accuracy = tf.assign(best_accuracy, np.mean(predictions))
                best_saver.save(sess, args.checkpoint_dir + '/best/' + args.checkpoint_name)


if __name__ == '__main__':
    args = args()
    main(args)
