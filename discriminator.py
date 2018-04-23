import tensorflow as tf


class Discriminator(object):
    """Generator

    Parameters
    ----------------------
    path: Path object
        Filewriterを作る場所を示すpath
    """

    def __init__(self,
                 bottom_width=28,
                 ch=1,
                 wscale=0.02,
                 is_training=None,
                 path=None):
        # register parameters
        self.bottom_width = bottom_width
        self.ch = ch
        self.wscale = wscale
        self.path = path
        self.is_training = is_training
        """
        with tf.Graph().as_default():
            tf.set_random_seed(20170311)  # 乱数を固定
            with tf.variable_scope("generator"):
                with tf.variable_scope("input"):
                    self.inputs = tf.placeholder(tf.float32, [None, 784])
                    self.labels = tf.placeholder(tf.float32, shape=[None, 10])

                logits = self.inference(self.inputs)
            # self.xentropy = self.losses(logits, self.labels)
            # self.train_step = self.training(self.xentropy)
            # self.accuracy = self.metrics(logits, self.labels)
            # tensorboard
            # self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # self.run_metadata = tf.RunMetadata()
            # tf.summary.scalar("loss", self.xentropy)
            # tf.summary.scalar("accuracy", self.accuracy)
            # self.summary = tf.summary.merge_all()
            # initialize
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            # make Filewriter
            self.writer_val = tf.summary.FileWriter(
                str(path / "val"), graph=self.sess.graph)
            # self.writer_train = tf.summary.FileWriter(
            #  str(path / "train"), graph=self.sess.graph)
        """

    def inference(self, inputs):
        """Function to perform inference processing

        Parameters
        ---------------------
        inputs: placeholder(shape=(n_batch, 28, 28, 1), tf.float32)
            input data.

        Return
        ---------------------
        logits: Tensor(shape=(n_batch, 1))
            output data not passing through tf.nn.sigmoid()
        """

        # define initializer
        # mean=0.0, stddev=1.0
        init = tf.initializers.truncated_normal(
            seed=20170311, stddev=self.wscale)

        # C-1
        outputs = tf.layers.conv2d(
            inputs=inputs,
            filters=64,
            kernel_size=5,
            strides=2,
            padding="same",
            activation=tf.nn.leaky_relu,
            kernel_initializer=init,
            bias_initializer=init,
            name="conv1")

        # C-2
        outputs = tf.layers.conv2d(
            inputs=outputs,
            filters=32,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=init,
            bias_initializer=init,
            name="conv2")
        # BN-2
        outputs = tf.layers.batch_normalization(
            inputs=outputs, training=self.is_training)
        # Activation-2
        outputs = tf.nn.leaky_relu(outputs)

        # C-3
        outputs = tf.layers.conv2d(
            inputs=outputs,
            filters=16,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=init,
            bias_initializer=init,
            name="conv3")
        # BN-3
        outputs = tf.layers.batch_normalization(
            inputs=outputs, training=self.is_training)
        # Activation-3
        outputs = tf.nn.leaky_relu(outputs)

        # FC-4
        logits = tf.layers.dense(
            inputs=outputs,
            units=1
        )
        return logits

    def losses(output, lable):
        # convert labels into one-hot labels
        one_hot = tf.one_hot(label, 10)


if __name__ == "__main__":
    import pathlib
    import numpy as np
    # import matplotlib as mpl
    # mpl.use('Agg')  # sshのために

    # import mnist data
    (x_train, label), _ = tf.keras.datasets.mnist.load_data()

    inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])
    is_training = tf.placeholder(tf.bool, [])

    # デレクトリの初期化
    path = "logs"
    abs_path = pathlib.Path(path).resolve()
    if abs_path.exists():
        try:
            abs_path.rmdir()
        except OSError:
            import shutil
            shutil.rmtree(abs_path)
        finally:
            print("Init Dir!!!")

    # mnist.train.next_batch に置いて中でnumpyのseedを使っているので...
    np.random.seed(20170311)
    sess = tf.Session()

    gen = Discriminator(is_training=is_training)
    gen.inference(inputs=inputs)
    writer = tf.summary.FileWriter(str(path), graph=sess.graph)
