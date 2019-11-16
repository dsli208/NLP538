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

        # Define input and output sets of biases and weights, as well as embeddings to be looked up
        stddev = math.sqrt(1/num_transitions)
        self.biases = tf.Variable(tf.random.truncated_normal([hidden_dim, 1]), trainable=True)
        self.biases2 = tf.Variable(tf.random.truncated_normal([num_transitions, 1]))
        self.weights1 = tf.Variable(tf.random.truncated_normal([hidden_dim, num_tokens * embedding_dim], mean=0.0, stddev=0.05), trainable=True) # tokens (features) * embedding_dim, hidden_dim
        self.weights2 = tf.Variable(tf.random.truncated_normal([num_transitions, hidden_dim], mean=0.0, stddev=0.05), trainable=True)
        self.embeddings  = tf.Variable(tf.random.truncated_normal([vocab_size, embedding_dim]), trainable=trainable_embeddings, dtype=tf.float32)

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
        # import pdb; pdb.set_trace()
        embeddings = tf.reshape(tf.nn.embedding_lookup(self.embeddings, inputs), [tf.shape(inputs)[0], self.embedding_dim * self.num_tokens]) # KEEP LINE # embedding dim x num tokens x batch size

        x = tf.add(tf.matmul(self.weights1, embeddings, transpose_a=False, transpose_b=True), self.biases)
        h = self._activation(x)
        logits_a = tf.add(tf.matmul(self.weights2, h), self.biases2)
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

        a = tf.constant(1, shape=logits.shape, dtype=tf.float32)
        b = tf.constant(0, shape=logits.shape, dtype=tf.float32)
        c = tf.constant(-999999, shape=logits.shape, dtype=tf.float32)
        # import pdb; pdb.set_trace()

        # Label_mask is applied for labels geq 0
        feasible_mask = tf.where(tf.greater_equal(labels, 0), a, b)
        correct_mask = tf.where(tf.greater_equal(labels, 1), a, b)

        # Apply masks accordingly - first mask label values less than 0, then values less than 1
        feasible_transitions = tf.multiply(logits, feasible_mask)
        correct_transitions = tf.multiply(feasible_transitions, correct_mask)
        tf_zero_mask = tf.where(tf.equal(correct_transitions, 0), b, a)

        x = tf.reduce_sum(correct_transitions)
        y = tf.reduce_sum(feasible_transitions)

        x = tf.reduce_mean(correct_transitions)
        y = tf.reduce_mean(feasible_transitions)

        e_x = tf.math.exp(x)
        e_y = tf.math.exp(y)

        feasible_sum = tf.add(e_x, e_y)

        feasible_transitions_log = tf.math.log(feasible_sum)
        correct_transitions_log = tf.math.log(e_x)

        loss = tf.subtract(feasible_transitions_log, correct_transitions_log)

        # Compute REGULARIZATION - Get l2 loss over all biases, weights, and embeddings
        bias_loss = tf.nn.l2_loss(self.biases)
        bias2_loss = tf.nn.l2_loss(self.biases2)
        w1_loss = tf.nn.l2_loss(self.weights1)
        w2_loss = tf.nn.l2_loss(self.weights2)
        embed_loss = tf.nn.l2_loss(self.embeddings)

        # SUM l2 loss over all biases, weights, and embeddings
        loss_sum_list = [bias_loss, w1_loss, w2_loss, embed_loss]

        # Multiply by lambda
        regularization = self._regularization_lambda * tf.math.add_n(loss_sum_list)

        # TODO(Students) End
        return loss + regularization
