import tensorflow as tf


class DCGAN(object):
    """DCGAN

    Parameters
    ----------------------
    path: Path object
        Filewriterを作る場所を示すpath
    """

    def __init__(self,
                 n_hidden=100,
                 bottom_width=4,
                 ch=128,
                 wscale=0.02,
                 is_training=None,
                 path=None):

        with tf.Graph().as_default():
            tf.set_random_seed(20170311)  # 乱数を固定
            with tf.variable_scope("model"):
                with tf.variable_scope("input"):
                    self.noise = tf.placeholder(
                        tf.float32, [None, 100], name="z")
                    self.image = tf.placeholder(
                        tf.float32, [None, 28, 28, 1], name="x")
                    self.is_training = tf.placeholder(
                        tf.bool, [], name="is_training")
                # generate fake image
                self.fake_image = self._generator(
                    self.noise,
                    self.is_training,
                    bottom_width=bottom_width,
                    ch=ch,
                    wscale=wscale)
                # define real loss and fake loss
                d_real = self._discriminator(self.image, self.is_training,
                                             wscale)
                d_fake = self._discriminator(
                    self.fake_image, self.is_training, wscale, reuse=True)

            # define generator loss and discriminator loss respectively
            self.loss_d, self.loss_g = self.losses(d_real, d_fake)
            # make train_op
            self.train_op = self.make_train_op(self.loss_d, self.loss_g)

            # tensorboard
            # self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # self.run_metadata = tf.RunMetadata()
            tf.summary.scalar("loss_d", self.loss_d)
            tf.summary.scalar("loss_g", self.loss_g)
            self.summary = tf.summary.merge_all()

            # initialize
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            # make Filewriter
            self.writer = tf.summary.FileWriter(
                str(path), graph=self.sess.graph)

    def _generator(self, inputs, is_training, wscale, bottom_width, ch):
        """build generator

        Parameters
        ---------------------
        inputs: placeholder(shape=(n_batch, n_dim=100), tf.float32)
            input data.

        is_training: placeholder(shape=(1), tf.bool)
            training flag.

        wscale: float
            initializer's stddev

        bottom_width: int
            Width when converting the output of the first layer
            to the 4-dimensional tensor

        ch: int
            Channel when converting the output of the first layer
            to the 4-dimensional tensor

        Return
        ---------------------
        fake_image: Tensor(shape=(n_batch, 1, 28, 28), tf.float32)
            range is -1 ~ 1
        """

        # define initializer
        # mean=0.0, stddev=1.0
        init = tf.initializers.truncated_normal(seed=20170311, stddev=wscale)

        with tf.variable_scope("generator", reuse=None):
            # FC-1
            outputs = tf.layers.dense(
                inputs=inputs,
                units=1024,
                kernel_initializer=init,
                bias_initializer=init,
                name="dense1")
            # BN-1
            outputs = tf.layers.batch_normalization(
                inputs=outputs, training=is_training)
            # Activation-1
            outputs = tf.nn.relu(outputs)

            # FC-2
            outputs = tf.layers.dense(
                inputs=outputs,
                units=bottom_width * bottom_width * ch,
                kernel_initializer=init,
                bias_initializer=init,
                name="dence2")
            # BN-2
            outputs = tf.layers.batch_normalization(
                inputs=outputs, training=is_training)
            # Activation-2
            outputs = tf.nn.relu(outputs)

            # reshape NHWC
            outputs = tf.reshape(outputs, [-1, bottom_width, bottom_width, ch])

            # Deconv-3
            outputs = tf.layers.conv2d_transpose(
                inputs=outputs,
                filters=ch // 2,
                kernel_size=3,
                strides=2,
                padding="same",
                kernel_initializer=init,
                bias_initializer=init,
                name="deconv3")
            # BN-3
            outputs = tf.layers.batch_normalization(
                inputs=outputs, training=is_training)
            # Activation-3
            outputs = tf.nn.relu(outputs)
            # Deconv-4
            outputs = tf.layers.conv2d_transpose(
                inputs=outputs,
                filters=ch // 4,
                kernel_size=4,
                strides=2,
                padding="same",
                kernel_initializer=init,
                bias_initializer=init,
                name="deconv4")
            # BN-4
            outputs = tf.layers.batch_normalization(
                inputs=outputs, training=is_training)
            # Activation-4
            outputs = tf.nn.relu(outputs)

            # Deconv-5
            fake_image = tf.layers.conv2d_transpose(
                inputs=outputs,
                filters=1,
                kernel_size=4,
                strides=2,
                padding="same",
                activation=tf.nn.tanh,
                kernel_initializer=init,
                bias_initializer=init,
                name="deconv5")

        return fake_image

    def _discriminator(self, inputs, is_training, wscale, reuse=None):
        """build discriminator

        Parameters
        ---------------------
        inputs: placeholder(shape=(n_batch, 28, 28, 1), tf.float32)
            input data.

        is_training: placeholder(shape=(1), tf.bool)
            training flag.

        wscale: float
            initializer's stddev

        reuse: boolean
            this parameter is used to tf.variable_scope()

        Return
        ---------------------
        logits: Tensor(shape=(n_batch, 1))
            output data not passing through tf.nn.sigmoid()
        """

        with tf.variable_scope("discriminator", reuse=reuse):
            # define initializer
            # mean=0.0, stddev=1.0
            init = tf.initializers.truncated_normal(
                seed=20170311, stddev=wscale)

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
                inputs=outputs, training=is_training)
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
                inputs=outputs, training=is_training)
            # Activation-3
            outputs = tf.nn.leaky_relu(outputs)

            # FC-4
            logits = tf.layers.dense(inputs=outputs, units=1)

        return logits

    def losses(output, dis_real, dis_fake):
        """define loss function

        Parameters
        -------------------
        dis_real: Tensor(shape=(num_batch, 1))
            logits of real image
        dis_fake: Tensor(shape=(num_batch, 1))
            logits of fake(generate) image
        
        Returns
        -----------------
        loss_d: Tensor(scalar)
            discriminator loss value to minimize

        loss_g; Tensor(salar)
            generator loss value to minimize

        """
        # convert labels into one-hot labels
        # one_hot = tf.one_hot(label, 10)

        # define loss function
        with tf.name_scope("losses"):
            with tf.name_scope("dis_loss"):
                loss_d_real = tf.losses.sigmoid_cross_entropy(
                    tf.ones_like(dis_real), dis_real)
                loss_d_fake = tf.losses.sigmoid_cross_entropy(
                    tf.zeros_like(dis_fake), dis_fake)
                loss_d = (loss_d_real + loss_d_fake) / 2
            with tf.name_scope("gen_loss"):
                loss_g = -tf.log(tf.clip_by_value(dis_fake, 1e-10, 1.0))

        return loss_d, loss_g

    def make_train_op(self, loss_d, loss_g):
        """make train_step

        Parameters
        ------------------
        loss_d: Tensor(scalar)
            discriminator loss value to minimize

        loss_g; Tensor(salar)
            generator loss value to minimize

        Return
        ------------------
        train_step: train_op
            If you execute this op learning progresses
        """

        with tf.name_scope("optimizer"):
            # extract trainable variables from generator and discriminator
            vars_g = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/generator')
            vars_d = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/discriminator')

            # print("vars_g", vars_g)
            # print("vars_d", vars_d)

            # It is necessary to update BN average_mean, etc ...
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                d_optim = tf.train.AdamOptimizer(
                    learning_rate=0.0002, beta1=0.5).minimize(
                        loss_d, var_list=vars_d)
                g_optim = tf.train.AdamOptimizer(
                    learning_rate=0.0002, beta1=0.5).minimize(
                        loss_g, var_list=vars_g)
                with tf.control_dependencies([g_optim, d_optim]):
                    train_op = tf.no_op(name='train')

        return train_op


if __name__ == "__main__":
    import pathlib
    import numpy as np
    # import matplotlib as mpl
    # mpl.use('Agg')  # sshのために

    # import mnist data
    # (x_train, label), _ = tf.keras.datasets.mnist.load_data()

    inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])
    is_training = tf.placeholder(tf.bool, [])

    # デレクトリの初期化
    path = "dcgan_logs"
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

    gan = DCGAN(path=path)