from dcgan import DCGAN
import matplotlib.pyplot as plt


def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(np.sqrt(total))
    rows = int(np.ceil(float(total) / cols))
    width, height = generated_images.shape[1:3]
    combined_image = np.zeros(
        (height * rows, width * cols), dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index / cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1)] =\
            image[:, :, 0]
    return combined_image


def plot_images(dcgan, path, epoch, encoder):
    np.random.seed(1)
    noise = np.random.uniform(-1, 1, (10 * 10, 100))
    labels = np.repeat(np.array([i for i in range(10)]), 10)
    labels = labels.reshape(-1, 1)
    one_hot = encoder.transform(labels).toarray()

    generated_image = dcgan.fake_image.eval(
        session=dcgan.sess,
        feed_dict={
            dcgan.noise: noise,
            dcgan.labels: one_hot,
            dcgan.is_training: False
        })
    np.random.seed()
    generated_image = generated_image * 127.5 + 127.5
    image = combine_images(generated_image)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.savefig(str(path / "epoch:{}.png".format(epoch + 1)))


if __name__ == "__main__":
    import pathlib
    import tensorflow as tf
    from sklearn.utils import shuffle
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np

    # import matplotlib as mpl
    # mpl.use('Agg')  # sshのために

    # mnist.train.next_batch に置いて中でnumpyのseedを使っているので...
    # np.random.seed(20170311)
    # import mnist data
    (train_image, labels), _ = tf.keras.datasets.mnist.load_data()
    train_image = train_image.reshape(-1, 28, 28, 1)
    train_image = (train_image - 127.5) / 127.5

    # one_hot
    labels = labels.reshape(-1, 1)
    encoder = OneHotEncoder()
    one_hot = encoder.fit_transform(labels).toarray()

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
    path = "image"
    abs_path2 = pathlib.Path(path).resolve()
    if abs_path2.exists():
        pass
    else:
        abs_path2.mkdir()

    # data
    batch_size = 128  # batch size
    n_batches = len(train_image) // batch_size  # number of batches
    epochs = 100  # number of epochs

    # build mlp
    dcgan = DCGAN(num_data=batch_size, path=abs_path)

    # earlystopping
    # early_stopping = EarlyStopping(patience=10, verbose=1)

    # history log
    logs = ["dis_loss", "gen_loss"]
    hist = {log: [] for log in logs}

    # training
    for epoch in range(epochs):
        train_image, one_hot = shuffle(train_image, one_hot)

        for i in range(n_batches):
            noise = np.random.uniform(-1, 1, (batch_size, 100))
            batch_image = train_image[i:i + batch_size]
            batch_label = one_hot[i:i + batch_size]
            # train dcgan
            dcgan.sess.run(
                dcgan.train_op,
                feed_dict={
                    dcgan.noise: noise,
                    dcgan.image: batch_image,
                    dcgan.labels: batch_label,
                    dcgan.is_training: True
                })

        # report loss
        noise = np.random.uniform(-1, 1, (len(train_image[:batch_size]), 100))
        dis_loss, gen_loss, summary = \
            dcgan.sess.run([dcgan.loss_d, dcgan.loss_g, dcgan.summary],
                           feed_dict={
                               dcgan.noise: noise,
                               dcgan.image: batch_image,
                               dcgan.labels: batch_label,
                               dcgan.is_training: False})
        print("epoch:{0:2d}  DLoss:{1:.5f} GLoss:{2:.5f}".format(
            epoch + 1, dis_loss, gen_loss))

        # draw image
        plot_images(dcgan, abs_path2, epoch, encoder)
        # write summary
        dcgan.writer.add_summary(summary, global_step=epoch)

    dcgan.sess.close()
