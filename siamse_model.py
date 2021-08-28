import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers


def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

  Arguments:
      margin: Integer, defines the baseline for distance for which pairs
              should be classified as dissimilar. - (default is 1).

  Returns:
      'constrastive_loss' function with data ('margin') attached.
  """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

      Arguments:
          y_true: List of labels, each label is of type float32.
          y_pred: List of predictions of same length as of y_true,
                  each label is of type float32.

      Returns:
          A tensor containing constrastive loss as floating point value.
      """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss


def create_siamese_mlp(shape, network_type, learning_rate=None, weights_path=None):
    input = layers.Input(shape)

    if network_type == "small":
        x = layers.Dense(256, activation="relu")(input)
    else:
        dense1 = layers.Dense(512, activation="relu")(input)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        x = layers.Dense(256)(dense2)
    embedding_network = Model(input, x)

    input_1 = layers.Input(shape)
    input_2 = layers.Input(shape)

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
    normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
    output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
    siamese = Model(inputs=[input_1, input_2], outputs=output_layer)

    if weights_path is not None:
        with open(weights_path, 'rb') as file:
            import pickle
            weights = pickle.load(file)
            siamese.set_weights(weights)

    opt = optimizers.Adam(lr=0.001, decay=1e-6)
    print(opt)
    if learning_rate is not None:
        siamese.compile(loss=loss(margin=1), optimizer=optimizers.SGD(learning_rate=learning_rate), metrics=["accuracy"])
    else:
        #siamese.compile(loss=loss(margin=1), optimizer="RMSprop", metrics=["accuracy"])
        siamese.compile(loss=loss(margin=1), optimizer=opt, metrics=["accuracy"])
    siamese.summary()
    return siamese


def create_mlp_embedding(siamese, index=None):
    functional_layer = siamese.get_layer(index=2)
    if index is None:
        index = len(functional_layer.layers) - 1
    index = min(index, len(functional_layer.layers) - 1) # get embedding from last function layer if given index is not valid.
    inter_output_model = Model(functional_layer.input, functional_layer.get_layer(index=index).output)
    return inter_output_model, index

