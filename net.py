import tensorflow as tf


class Generator(object):
    """Generator

    Model
    ---------------------
    FullyConnectLayer-1
        number of units: 256
    FullyConnectLayer-2
        number of units: 256
    FullyConnectLayer-3 (Output Layer)
        number of units: 10

    Parameters
    ----------------------
    path: Path object
        Filewriterを作る場所を示すpath
    """

    def __init__(self, n_hidden=100, bottom_width=4, ch=128, wsale=0.02, is_training, path):
        # register parameters
        self.n_hidden = n_hidden
        self.bottom_width = bottom_width
        self.ch = ch
        self.wscale = wscale
        self.path = path
        self.is_training = is_training
        
        with tf.Graph().as_default():
            tf.set_random_seed(20170311)  # 乱数を固定
            with tf.variable_scope("generator"):
                with tf.variable_scope("input"):
                    self.inputs = tf.placeholder(tf.float32, [None, 784])
                    self.labels = tf.placeholder(tf.float32, shape=[None, 10])

                logits = self.inference(self.inputs)
            self.xentropy = self.losses(logits, self.labels)
            self.train_step = self.training(self.xentropy)
            self.accuracy = self.metrics(logits, self.labels)
            # tensorboard
            # self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # self.run_metadata = tf.RunMetadata()
            tf.summary.scalar("loss", self.xentropy)
            tf.summary.scalar("accuracy", self.accuracy)
            self.summary = tf.summary.merge_all()
            # initialize
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            # make Filewriter
            self.writer_val = tf.summary.FileWriter(str(path / "val"),
                                                    graph=self.sess.graph)
            self.writer_train = tf.summary.FileWriter(str(path / "train"),
                                                      graph=self.sess.graph)

    def inference(self, inputs):
        """Function to perform inference processing

        Parameters
        ---------------------
        inputs: placeholder(shape=(n_batch, n_dim=100), tf.float32)
            input data.

        Return
        ---------------------
        fake_image: Tensor(shape=(n_batch, 1, 28, 28), tf.float32)
            range is -1 ~ 1
        """

        # define initializer
        # mean=0.0, stddev=1.0
        init = tf.initializers.truncated_normal(seed=20170311,
                                                     stddev=self.wscale)

        # FC-1
        outputs = tf.layers.dense(inputs=inputs,
                                  units=1024,
                                  kernel_initializer=init,
                                  bias_initializer=init,
                                  name="dense1")
        # BN-1
        outputs = tf.layers.batch_normalization(inputs=outputs,
                                                training=self.is_training)
        # Activation-1
        outputs = tf.nn.relu(outputs)

        # FC-2
        outputs = tf.layers.dense(inputs=outputs,
                                  units=self.bottom_width*self.bottom_width*self.ch,
                                  kernel_initializer=init,
                                  bias_initializer=init,
                                  name="dence2")
        # BN-2
        outputs = tf.layers.batch_normalization(inputs=outputs,
                                                training=self.is_training)
        # Activation-2
        outputs = tf.nn.relu(outputs)

        # reshape NHWC
        outputs = tf.reshape(outputs, [-1, self.bottom_width, self.bottom_width])

        # Deconv-3
        outputs = tf.layers.conv2d_transpose(inputs=outputs,
                                             filters=self.ch//2,
                                             kernel_size=3,
                                             strides=2,
                                             padding="same",
                                             kernel_initializer=init,
                                             bias_initializer=init,
                                             name="deconv3")
        # BN-3
        outputs = tf.layers.batch_normalization(inputs=outputs,
                                                training=self.is_training)
        # Activation-3
        outputs = tf.nn.relu(outputs)
        # Deconv-4
        outputs = tf.layers.conv2d_transpose(inputs=outputs,
                                             filters=self.ch//4,
                                             kernel_size=4,
                                             strides=2,
                                             padding="same",
                                             kernel_initializer=init,
                                             bias_initializer=init,
                                             name="deconv4")
        # BN-4
        outputs = tf.layers.batch_normalization(inputs=outputs,
                                                training=self.is_training)
        # Activation-4
        outputs = tf.nn.relu(outputs)

        # Deconv-5
        fake_image = tf.layers.conv2d_transpose(inputs=outputs,
                                                filters=1,
                                                kernel_size=4,
                                                strides=2,
                                                padding="same",
                                                activation=tf.nn.tanh
                                                kernel_initializer=init,
                                                bias_initializer=init,
                                                name="deconv5")
        
        return fake_image
