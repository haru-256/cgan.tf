from dcgan import DCGAN


if __name__ == "__main__":
    import pathlib
    import numpy as np
    import tensorflow as tf
    # import matplotlib as mpl
    # mpl.use('Agg')  # sshのために

    # mnist.train.next_batch に置いて中でnumpyのseedを使っているので...
    np.random.seed(20170311)
    # import mnist data
    (image, _), _ = tf.keras.datasets.mnist.load_data()
    image = image.reshape(-1, 28 ,28, 1)

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

    # data
    batch_size = 128  # batch size
    n_batches = len(image) // batch_size  # number of batches
    epochs = 100  # number of epochs

    # build mlp
    dcgan = DCGAN(path=path)

    # earlystopping
    # early_stopping = EarlyStopping(patience=10, verbose=1)

    # history log
    logs = ["dis_loss", "gen_loss"]
    hist = {log: [] for log in logs}

    for epoch in range(epochs):
        # training
        for i in range(n_batches):
            noise = np.random.uniform(-1, 1, (batch_size, 100))
            

            # train dcgan
            dcgan.sess.run(
                dcgan.train_op,
                feed_dict={
                    dcgan.noise: noise,
                    dcgan.image: batch_xs,
                    dcgan.is_training: True
                })

        # report loss
        dis_loss, gen_loss, summary = \
            dcgan.sess.run([dcgan.loss_d, dcgan.loss_g, dcgan.summary],
                           feed_dict={
                               dcgan.noise: noise,
                               dcgan.image: batch_xs,
                               dcgan.is_training: True})
        print("epoch:{0:2d}  DLoss:{1:.5f} GLoss:{2:.5f}".format(
                  epoch + 1, dis_loss, gen_loss))
        hist["dis_loss"].append(dis_loss)
        hist["gen_loss"].append(gen_loss)

        # write summary
        dcgan.writer.add_summary(summary, global_step=epoch)

    dcgan.sess.close()
