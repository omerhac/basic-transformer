import tensorflow as tf
import numpy as np
import modules
import preprocess
import time


class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule as described in the paper"""
    def __init__(self, d_model, warmup_steps=4000):
        super(TransformerSchedule, self).__init__()

        self._d_model = tf.cast(d_model, tf.float32)
        self._warmup_steps = warmup_steps

    def __call__(self, step):
        learning_rate = tf.pow(self._d_model, -0.5) * tf.minimum(
            step**-0.5,
            step * self._warmup_steps**-1.5
        )
        return learning_rate


def load_checkpoint(transformer, optimizer=None, load_dir='checkpoints'):
    """Load transformer model and optimizer from load dir. Return checkpoints manager"""
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, load_dir, max_to_keep=10)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    return manager


def train_transformer(dataset, transformer=None, epochs=20, load_dir='checkpoints'):
    if not transformer:
        transformer = modules.Transformer(8002, 8002, 40,
                                          d_model=128, d_forward_layer=512,
                                          attention_heads=8, n_layers=4)

    # create optimizer
    lr_schedule = TransformerSchedule(transformer._d_model)  # d_model
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=10e-9)

    # load checkpoints
    manager = load_checkpoint(transformer, optimizer, load_dir=load_dir)

    # aggregators
    loss_agg = tf.keras.metrics.Mean(name='mean_loss')
    accuracy_agg = tf.keras.metrics.SparseCategoricalAccuracy(name='mean_accuracy')
    curr_time = time.time()  # timing the procedure for optimizing performance

    # define loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')

    @tf.function
    def loss_function(y_true, y_pred):
        """Masked categorical crossentropy over target language labels"""
        loss_ = loss_object(y_true, y_pred)
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        mask = tf.cast(mask, loss_.dtype)
        loss = loss_ * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    # def train step
    train_sig = (
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64)
    )

    @tf.function(input_signature=train_sig)
    def train_step(encoder_input, target):
        """Make one train step
        Args:
            encoder_input: input language tokens
            target: target language tokens
        """

        target_true = target[:, 1:]  # the true labels, given for evaluating loss
        target_inp = target[:, :-1]  # the true labels, shifted right to feed the model as if it were its predictions

        with tf.GradientTape() as tape:
            model_predictions = transformer(encoder_input, target_inp, training=True)  # forward pass

            # calculate loss
            loss = loss_function(target_true, model_predictions)

            # apply gradients
            grads_and_vars = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(grads_and_vars, transformer.trainable_variables))

            # aggregate accuracy
            accuracy_agg(target_true, model_predictions)

        return loss

    # train loop
    for epoch in range(epochs):
        print("EPOCH number: {}".format(epoch))

        for batch_num, (inp, tar) in enumerate(dataset):
            loss = train_step(inp, tar)

            # append values
            loss_agg(loss)

            if batch_num % 50 == 0:
                print("Batch {}".format(batch_num))
                print("Average loss {}, Average Accuracy {}. Time taken: {}".format(loss_agg.result(),
                                                                                    accuracy_agg.result(),
                                                                                    time.time() - curr_time))
                curr_time = time.time()

        if epoch % 5 == 0:
            save_path = manager.save()
            print("Saved model after {} epochs to {}".format(epoch, save_path))


if __name__ == '__main__':
    train_transformer(preprocess.get_transformer_datasets(64, 40, 20000)[0], load_dir='checkpoints/portuguese-english')
    #train_transformer(preprocess.get_transformer_datasets(10, 10, 100)[0], load_dir='checkpoints/portuguese-english')