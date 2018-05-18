import tensorflow as tf
import numpy as np

def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    o_h = np.zeros(n)
    o_h[x] = 1.
    return o_h

num_classes = 3
batch_size = 4

# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(i, num_classes)  # [one_hot(float(i), num_classes)]
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch

# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 18 * 33 * 64]), units=50, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y

example_batch_train, label_batch_train = dataSource(["Mi_DataSet/train/huchaGRAY/*.jpg", "Mi_DataSet/train/botellaGRAY/*.jpg", "Mi_DataSet/train/enchufeGRAY/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["Mi_DataSet/valid/huchaGRAY/*.jpg", "Mi_DataSet/valid/botellaGRAY/*.jpg", "Mi_DataSet/valid/enchufeGRAY/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["Mi_DataSet/test/huchaGRAY/*.jpg", "Mi_DataSet/test/botellaGRAY/*.jpg", "Mi_DataSet/test/enchufeGRAY/*.jpg"], batch_size=batch_size)

label_batch_train = tf.cast(label_batch_train, tf.float32)
label_batch_valid = tf.cast(label_batch_valid, tf.float32)
label_batch_test = tf.cast(label_batch_test, tf.float32)

example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - label_batch_train))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - label_batch_valid))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    valError0=0
    valError1=0
    a_trainError = []
    a_valError = []

    for _ in range(430):
        sess.run(optimizer)

        trainError = sess.run(cost)
        a_trainError.append(trainError)
        actualValError = sess.run(cost_valid)
        a_valError.append(actualValError)

        print("Iter:", _, "---------------------------------------------")
        print("Error de validacion:", actualValError)

        if _ == 0:
            valError0;valError1 = actualValError
        else:
            valError0=valError1
            valError1=actualValError

        if abs(valError1 - valError0) / valError0 < 0.00001:
            break

    print("------- Resultados Finales -------")
    print("Epoch: #", _)
    print("Error de Entrenamiento Final --->", trainError)
    print("Error de Validacion Final ---> ", valError1)

    test_result = sess.run(example_batch_test_predicted)

    aciertos = 0

    for label, nn in zip(label_batch_test.eval(), test_result):
        if np.argmax(nn) == np.argmax(label):
            aciertos += 1

    precision = aciertos / len(label_batch_test.eval()) * 100
    print("Accuracy :", precision, "%")

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    # Setup and Show the Figures
    import matplotlib.pyplot as plt

    plt.ioff()
    plt.subplot(1, 2, 1)
    plt.plot(a_trainError)
    plt.xlabel("Time (epochs)")
    plt.ylabel("Error")
    plt.title("Error de Entrenamiento")

    plt.subplot(1, 2, 2)
    plt.xlabel("Time (epochs)")
    plt.ylabel("Error")
    plt.title("Error de Validacion")
    plt.plot(a_valError, 'g')

    plt.savefig("figuras/Figura2.jpeg")
    plt.show()

    coord.request_stop()
    coord.join(threads)

