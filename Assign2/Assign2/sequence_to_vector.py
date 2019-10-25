# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models


class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """

        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start

        # self.input_dim = input_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.dense_layers = []

        for i in range(num_layers):
            self.dense_layers.append(tf.keras.layers.Dense(input_dim, activation='relu'))

        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor, # Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
             sequence_mask: tf.Tensor, # Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that token is a real token, and 0 indicate that it's a padding token.
             training=False) -> tf.Tensor:
        # TODO(students): start

        if training:
            # Drop out - want to turn off some neurons in the NN
            # Some inputs we want to ignore
            # Which ones are 0's in the matrix and which ones are 1's?

            a = tf.constant(1, shape=vector_sequence.shape, dtype=tf.float32)
            b = tf.constant(0, shape=vector_sequence.shape, dtype=tf.float32)

            uni_mask = tf.Variable(tf.random.uniform(tf.shape(vector_sequence), minval=0, maxval=1, dtype=tf.float32, seed=self.dropout))
            bernoulli_mask = tf.where(tf.greater_equal(uni_mask, self.dropout), a, b)
            # bernoulli_mask = tf.map_fn(lambda x: tf.cond(x >= self.dropout, lambda x: 1, lambda x: 0), uni_mask)

            # tf.print(uni_mask)
            # tf.print(bernoulli_mask)
            # bernoulli_mask = tf.reshape(bernoulli_mask, vector_sequence)

            inputs = tf.multiply(vector_sequence, bernoulli_mask)

            inputs = tf.multiply(inputs, tf.expand_dims(sequence_mask, 2))
        else:
            inputs = vector_sequence

        # Get average
        # import pdb
        # pdb.set_trace()
        h = tf.reduce_mean(inputs, axis=1) # avg

        for d in self.dense_layers:
            h = d(h)

        combined_vector = tf.nn.softmax(h)
        layer_representations = h

        # TODO(students): end
        return {"combined_vector": combined_vector, "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)

        # TODO(students): start
        self.num_layers = num_layers

        self.gru_layers = []
        for i in range(0, num_layers - 1):
            self.gru_layers.append(tf.keras.layers.GRU(input_dim, activation='tanh', return_state=True, return_sequences=True))
        self.gru_layers.append(tf.keras.layers.GRU(input_dim, activation='tanh', return_state=True, return_sequences=False))
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        # if training:
            # sequence_mask_2 = tf.expand_dims(sequence_mask, 2)
            # inputs = tf.multiply(vector_sequence, sequence_mask_2)
        # else:
            # inputs = vector_sequence
        layer_representations_list = []
        sequence_mask_2 = tf.expand_dims(sequence_mask, 2)
        inputs = tf.multiply(vector_sequence, sequence_mask_2)
        h = inputs
        for g in self.gru_layers:
            # h = tf.expand_dims(h[0], [1])
            # import pdb
            # pdb.set_trace()
            h = g(h, mask=sequence_mask_2)
            layer_representations_list.append(h)


        combined_vector = layer_representations_list[-1][0]
        layer_representations = h

        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
