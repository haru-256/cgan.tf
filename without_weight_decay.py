import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class MLP(object):
    """Multi Layer Perceptron

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

    def __init__(self, path):
        with tf.Graph().as_default():
            tf.set_random_seed(20170311)  # 乱数を固定
            with tf.variable_scope("model"):
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
        """推論処理を行う

        Parameters
        ---------------------
        inputs: placeholder(shape=(n_batch, n_dim=784), tf.float32)
            input data.

        Return
        ---------------------
        logits: Tensor(shape=(n_batch, n_labels))
            output data without tf.nn.softmax
        """

        # define initializer
        # He's initialization (uniform)
        he_normal = tf.initializers.variance_scaling(scale=2.0,
                                                     mode="fan_in",
                                                     distribution="uniform",
                                                     seed=20170311)
        # mean=0.0, stddev=1.0
        bias_init = tf.initializers.truncated_normal(seed=20170311)

        # FC-1
        outputs = tf.layers.dense(inputs=inputs,
                                  units=256,
                                  activation=tf.nn.relu,
                                  kernel_initializer=he_normal,
                                  bias_initializer=bias_init,
                                  name="dense1")
        # FC-2
        outputs = tf.layers.dense(inputs=outputs,
                                  units=256,
                                  activation=tf.nn.relu,
                                  kernel_initializer=he_normal,
                                  bias_initializer=bias_init,
                                  name="dence2")
        # Output Layer
        logits = tf.layers.dense(inputs=outputs,
                                 units=10,
                                 activation=tf.nn.relu,
                                 kernel_initializer=he_normal,
                                 bias_initializer=bias_init,
                                 name="output")

        return logits

    def losses(self, logits, labels):
        """誤差関数の構築

        Parameters
        -------------------
        logits: inferenceの返り値

        labels: onehot label

        Return
        ------------------
        xentropy: cross entropy
        """

        with tf.name_scope("losses") as scope:
            xentropy = tf.losses.softmax_cross_entropy(logits=logits,
                                                       onehot_labels=labels,
                                                       scope=scope)

        return xentropy

    def training(self, loss):
        """train_stepを構築する関数

        Parameters
        --------------------
        loss: lossの返り血
        """
        with tf.name_scope("optimizer"):
            global_step = tf.get_variable("global_step",
                                          initializer=0,
                                          trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_step = optimizer.minimize(loss,
                                            global_step=global_step)

        return train_step

    def metrics(self, logits, labels):
        """metricsを返す関数

        Parameters
        -------------------
        logits: inferenceの返り値

        labels: onehot label

        Return
        ------------------
        accuracy: Percentage of correct answers
        """
        with tf.name_scope("metrics"):
            correct_prediction = tf.equal(tf.argmax(logits, axis=1),
                                          tf.argmax(labels, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy


# Earlystoppingを行うクラス
class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0  # 連続して誤差が増加すると終了するため常に初期化
            self._loss = loss

        return False


if __name__ == "__main__":
    import pathlib
    import numpy as np
    import matplotlib as mpl
    mpl.use('Agg')  # sshのために
    import matplotlib.pyplot as plt

    # デレクトリの初期化
    path = "without_logs"
    abs_path = pathlib.Path(path).resolve()
    if abs_path.exists():
        try:
            abs_path.rmdir()
        except OSError:
            import shutil
            shutil.rmtree(abs_path)
        finally:
            print("Init Dir!!!")

    # data download
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    # mnist.train.next_batch に置いて中でnumpyのseedを使っているので...
    np.random.seed(20170311)

    # data
    batch_size = 100  # batch size
    n_batches = len(mnist.train.labels) // batch_size  # number of batches
    epochs = 200  # number of epochs

    # build mlp
    mlp = MLP(path=abs_path)

    # earlystopping
    # early_stopping = EarlyStopping(patience=10, verbose=1)

    # history log
    logs = ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]
    hist = {log: [] for log in logs}

    for epoch in range(epochs):
        # training
        for i in range(n_batches):
            batch_xs, batch_ts = mnist.train.next_batch(batch_size=batch_size)
            mlp.sess.run(
                mlp.train_step, feed_dict={
                    mlp.inputs: batch_xs,
                    mlp.labels: batch_ts}
            )

        # validate
        train_loss, train_accuracy, t_summary = \
            mlp.sess.run([mlp.xentropy, mlp.accuracy, mlp.summary],
                         feed_dict={
                             mlp.inputs: mnist.train.images,
                             mlp.labels: mnist.train.labels})
        val_loss, val_acurracy, v_summary = \
            mlp.sess.run([mlp.xentropy, mlp.accuracy, mlp.summary],
                         feed_dict={
                             mlp.inputs: mnist.validation.images,
                             mlp.labels: mnist.validation.labels})
        print("epoch:{0:2d}  TLoss:{1:.5f} VLoss:{2:.5f}"
              " TAccuracy:{3:.5f} VAccuracy:{4:.5f}".format(epoch+1,
                                                            train_loss,
                                                            val_loss,
                                                            train_accuracy,
                                                            val_acurracy))
        hist["val_loss"].append(val_loss)
        hist["train_loss"].append(train_loss)
        hist["val_accuracy"].append(val_acurracy)
        hist["train_accuracy"].append(train_accuracy)

        # get global_step
        global_step_tensor = tf.train.get_global_step(graph=mlp.sess.graph)
        global_step = tf.train.global_step(sess=mlp.sess,
                                           global_step_tensor=global_step_tensor)
        # write summary
        mlp.writer_train.add_summary(t_summary,
                                     global_step=global_step)
        mlp.writer_val.add_summary(v_summary,
                                   global_step=global_step)
        # Early Stopping check
        # if early_stopping.validate(val_loss):
        #    break

    test_loss, test_acurracy = \
        mlp.sess.run([mlp.xentropy, mlp.accuracy],
                     feed_dict={
                         mlp.inputs: mnist.test.images,
                         mlp.labels: mnist.test.labels})
    print("Test Loss:{0:.5f} "
          "Test Accuracy{1:.5f}".format(test_loss, test_acurracy))
    # Result: Test Loss:0.24164 Test Accuracy0.98220
    mlp.sess.close()

    # plot history
    plt.style.use("seaborn-paper")
    """
    matplotlib.rcParams['text.latex.preamble'] = \
        [r"\\usepackage{amsmath}", r"\\usepackage{bm}"]
    matplotlib.rc('text', usetex=True)
    """
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.set_title("Weight Decay")
    ax1.plot(hist["val_loss"], label="val loss", color="forestgreen", ls="--")
    ax1.plot(hist["train_loss"], label="train loss", color="firebrick", ls="--")
    ax1.set_ylabel('Loss')
    ax1.set_xlabel("Step")
    ax1.legend(shadow=True, frameon=True, edgecolor='k',
               bbox_to_anchor=(1.05, -0.1), loc='lower left', borderaxespad=0)

    ax2 = ax1.twinx()
    ax2.plot(hist["val_accuracy"], label="val accuracy", color="forestgreen")
    ax2.plot(hist["train_accuracy"], label="train accuracy", color="firebrick")
    ax2.set_ylabel("Accuracy")
    ax2.legend(shadow=True, frameon=True, edgecolor='k',
               bbox_to_anchor=(1.05, 1.03), loc='upper left', borderaxespad=0)

    fig.tight_layout()
    fig.savefig("without_weight_decay.pdf")
