import tensorflow as tf
import numpy as np
import modules
import preprocess

class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule as described in the paper"""
    def __init__(self, d_model, warmup_steps=4000):
        super(TransformerSchedule, self).__init__()

        self._d_model = tf.cast(d_model, tf.float32)
        self._warmup_steps = warmup_steps

    def __call__(self, step):
        learning_rate = tf.pow(self._d_model, -0.5) * tf.minimum(
            tf.pow(step, -0.5), step * tf.pow(self._warmup_steps, -1.5)
        )
        return learning_rate


@tf.function
def loss_function(y_true, y_pred):
    """Masked categorical crossentropy over target language labels"""
    loss_ = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')(y_true, y_pred)
    mask = modules.pad_mask(y_true)
    loss = loss_ * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def train_step(inp, target, model, optimizer):
    """Make one train step
    Args:
        inp: input language tokens
        target: target language tokens
        model: transformer model to train
        optimizer: optimizer to apply gradients
    """

    target_true = target[1:, :]  # the true labels, given for evaluating loss
    target_inp = target[:-1, :]  # the true labels, shifted right to feed the model as if it were its predictions

    with tf.GradientTape() as tape:
        model_predictions = model(inp, target_inp, training=True)  # forward pass

        # calculate loss
        loss = loss_function(target_true, model_predictions)

        # apply gradients
        grads_and_vars = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_grads(zip(model.trainable_variables, grads_and_vars))

    return loss


def train_transformer(dataset, transformer=None, epochs=20, save_dir='checkpoints'):
    if not transformer:
        transformer = modules.Transformer(8500, 8000)

    # create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=TransformerSchedule, beta_1=0.9, beta_2=0.98, epsilon=10e-9)

    # load checkpoints
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, save_dir, max_to_keep=10)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    # train loop
    for epoch in range(epochs):
        print("EPOCH number: {}".format(epoch))

        for batch_num, (inp, target) in enumerate(dataset):
            loss = train_step(inp, target, transformer, optimizer)

            print("Batch number {} loss is: {}".format(loss.numpy()))

        if epoch % 5 == 0:
            save_path = manager.save()
            print("Saved model after {} epochs to {}".format(epochs, save_path))


if __name__ == '__main__':
    train_transformer(preprocess.get_transformer_datasets(64, 40, 100)[0])