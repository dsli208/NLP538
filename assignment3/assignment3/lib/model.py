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
        self.weights1 = tf.Variable(tf.random.truncated_normal([hidden_dim, num_tokens * embedding_dim]), trainable=True) # tokens (features) * embedding_dim, hidden_dim
        self.weights2 = tf.Variable(tf.random.truncated_normal([num_transitions, hidden_dim]), trainable=True)
        self.embed_array  = tf.Variable(tf.random.truncated_normal([vocab_size, embedding_dim]), trainable=True)
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
        embeddings = tf.reshape(tf.nn.embedding_lookup(self.embed_array, inputs), [tf.shape(inputs)[0], self.embedding_dim * self.num_tokens])
        # embeddings = tf.reshape(tf.nn.embedding_lookup(self.embed_array, inputs), [self.embedding_dim, self.num_tokens, tf.shape(inputs)[0]]) # embedding dim x num tokens x batch size
        # print("Model called")

        x = tf.add(tf.matmul(self.weights1, embeddings, transpose_a=False, transpose_b=True), self.biases)
        h = self._activation(x)
        logits_a = tf.matmul(self.weights2, h)
        logits = tf.transpose(logits_a)

        # import pdb; pdb.set_trace()

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

        label_mask = tf.where(tf.greater_equal(labels, 0), a, b)

        p = tf.nn.softmax(logits)
        logits_a = tf.multiply(tf.math.log(p), label_mask)
        logits_arr = tf.reduce_sum(logits_a, 1)
        loss = tf.reduce_mean(logits_arr)

        regularization_a = tf.multiply(self.regularization_lambda, self.weights1)
        regularization_arr = tf.reduce_sum(regularization_a, 1)
        regularization = tf.reduce_mean(regularization_arr)

        # TODO(Students) End
        return loss + regularization
