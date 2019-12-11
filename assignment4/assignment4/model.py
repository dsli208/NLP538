import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS

class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        # Look at the GRU from HW2 - make it BIDIRECTIONAL
        self.gru = tf.keras.layers.GRU(hidden_size, activation='tanh', return_sequences=True)
        self.bidirectional = tf.keras.layers.Bidirectional(self.gru)
        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        # Literally just wrapping a GRU layer
        tanh_layer = tf.keras.activations.tanh
        m = tanh_layer(rnn_outputs)
        softmax_input = tf.matmul(self.omegas, m, transpose_a=True, transpose_b=True)
        softmax = tf.nn.softmax(softmax_input)
        r = tf.matmul(rnn_outputs, softmax, transpose_a=True, transpose_b=True)
        sentence_rep = tanh_layer(r)
        output = tf.squeeze(sentence_rep)

        ### TODO(Students) END

        return output

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START
        embed_combined = tf.concat([word_embed, pos_embed], 2)
        rnn_outputs = self.bidirectional(embed_combined)
        attention = self.attn(rnn_outputs)
        logits = self.decoder(attention)
        ### TODO(Students) END

        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START
        # ...
        ### TODO(Students END

    def call(self, inputs, pos_inputs, training):
        raise NotImplementedError
        ### TODO(Students) START
        # ...
        ### TODO(Students END
