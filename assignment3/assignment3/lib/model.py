# inbuilt lib imports:
from typing import Dict
import math

# external libs
import tensorflow as tf
from tensorflow.keras import models, layers

# project imports


class CubicActivation(layers.Layer):
    """
    Cubic activation as described in the paper.
    """
    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        vector : ``tf.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start
        # Comment the next line after implementing call.
        return tf.pow(vector, 3)
        # raise NotImplementedError
        # TODO(Students) End


class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``int``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students) Start

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.num_transitions = num_transitions
        self.regularization_lambda = regularization_lambda
        self.trainable_embeddings = trainable_embeddings

        # Biases, weights
        # Defaults for embeddings
        # Define a matrix that's the shape that you need for the multiplication with inputs
        self.biases = tf.Variable(tf.random.truncated_normal([hidden_dim, 1]), trainable=True)
        self.weights1 = tf.Variable(tf.random.truncated_normal([hidden_dim, num_tokens * embedding_dim], mean=0.0, stddev=0.05), trainable=True) # tokens (features) * embedding_dim, hidden_dim
        self.weights2 = tf.Variable(tf.random.truncated_normal([num_transitions, hidden_dim], mean=0.0, stddev=0.05), trainable=True)
        self.embed_array  = tf.Variable(tf.random.truncated_normal([vocab_size, embedding_dim]), trainable=trainable_embeddings)
        # Embeddings = tf.nn.embedding_lookup
        # Generate them = tf.Variable(tf.random.truncated_normal(vocab_size, embedding_dim))

        # TODO(Students) End

    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        #
        # TODO(Students) Start
        # vocab x embedding
        # print(self.embed_array)
        self.embeddings = tf.reshape(tf.nn.embedding_lookup(self.embed_array, inputs), [tf.shape(inputs)[0], self.embedding_dim * self.num_tokens]) # KEEP LINE # KEEP LINE
        # self.embeddings = tf.transpose(tf.reshape(tf.nn.embedding_lookup(self.embed_array, inputs), [self.embedding_dim * self.num_tokens, tf.shape(inputs)[0]]))
        # embeddings = tf.reshape(tf.nn.embedding_lookup(self.embed_array, inputs), [self.embedding_dim, self.num_tokens, tf.shape(inputs)[0]]) # embedding dim x num tokens x batch size
        # import pdb; pdb.set_trace()

        x = tf.add(tf.matmul(self.weights1, self.embeddings, transpose_a=False, transpose_b=True), self.biases)
        h = self._activation(x)
        logits_a = tf.matmul(self.weights2, h)
        logits = tf.transpose(logits_a)

        # TODO(Students) End
        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict

    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:
        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start

        # logits is shape 91 x 10000, labels 10000 x 91, so transpose the latter (labels)
        labels_t = tf.transpose(labels)

        # use labels to create a mask (exclude values where associated value in labels is -1, keep values that are 0 or 1)
        a = tf.constant(1, shape=logits.shape, dtype=tf.float32)
        b = tf.constant(0, shape=logits.shape, dtype=tf.float32)
        # a = tf.ones_like(labels)
        # b = tf.zeros_like(labels)

        label_mask = tf.where(tf.greater_equal(labels, 0), a, b)
        logits_mask_1 = tf.where(tf.greater_equal(logits, 0), a, b)
        logits_mask_2 = tf.where(tf.greater_equal(labels, 1), a, b)
        label_mask_f = tf.dtypes.cast(label_mask, tf.float32)
        logits_mask_2f = tf.dtypes.cast(logits_mask_2, tf.float32)

        masked_logits = tf.multiply(logits, label_mask_f)
        softmax = tf.nn.softmax(masked_logits)
        p = tf.math.log(softmax + 1.0e-10) # include 0 and 1 labels
        logits_a = tf.multiply(p, logits_mask_2f)
        logits_arr = tf.reduce_sum(logits_a, 1) # ONLY include 1 label
        loss = tf.math.negative(tf.reduce_mean(logits_arr))

        loss_vec = tf.nn.softmax_cross_entropy_with_logits((labels >= 0) * labels, logits)
        # print(loss_vec)

        # loss = tf.reduce_mean(loss_vec)
        # print(loss)

        # regularization_a = tf.multiply(self.regularization_lambda, self.weights1)
        # regularization_arr = tf.reduce_sum(regularization_a, 1)
        # regularization = tf.reduce_mean(regularization_arr)

        bias_loss = tf.nn.l2_loss(self.biases)
        w1_loss = tf.nn.l2_loss(self.weights1)
        w2_loss = tf.nn.l2_loss(self.weights2)
        embed_loss = tf.nn.l2_loss(self.embeddings)

        loss_sum_list = [bias_loss, w1_loss, w2_loss, embed_loss]

        regularization = self._regularization_lambda * tf.math.add_n(loss_sum_list)

        # import pdb; pdb.set_trace()

        # TODO(Students) End
        return loss + regularization
