import tensorflow as tf
import sys
import math

def cross_entropy_loss(inputs, true_w):
    # Start thinking in terms of matrices, not single vectors
    # Each vector in batch is "u"
    # Matrix multiplication or value multiplication?

    print("Print inputs")

    x = tf.Print(inputs, [inputs], "DEBUG inputs: ")
    y = tf.Print(true_w, [true_w], "DEBUG true_w: ")
    # tf.compat.v1.enable_eager_execution()
    # with tf.Session() as s1:
        # t1 = tf.print(inputs, output_stream = sys.stdout)
        # t2 = tf.print(true_w, output_stream = sys.stdout)
        # s1.run([x])
        # s1.run([y])
    # t1.eval()
    # t2.eval()
    print("Printed")

    # A = log(exp({u_o}^T v_c))
    # Use indexing to get the word we want
    # Multiply matrices, take one vector
    dot1 = tf.multiply(inputs, tf.transpose(true_w))
    print(dot1)
    a_vec = tf.diag_part(dot1)

    A = tf.log(tf.exp(a_vec))

    # exp, then sum, then log
    # B = log(\sum{exp({u_w}^T v_c)})
    # Multiply matrices
    # dot2 = tf.multiply(inputs, tf.transpose(true_w))
    b1 = tf.exp(dot1)
    b_vec = tf.reduce_sum(b1, 1, keep_dims=True)
    B = tf.log(b_vec)

    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    print("End of cross entropy loss")
    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size]. (tensor)
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1]. (tensor)
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled]. (not a tensor)
    unigram_prob: Unigram probability. Dimesion is [Vocabulary]. (not a tensor)

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

    u_dot = tf.multiply(tf.transport(inputs), labels)
    u_o = tf.diag_part(u_dot)
    u_x = tf.reduce_sum(u_dot, 1, keep_dims=True)

    b_o = tf.gather(biases, labels) # [biases[labels[i]] for i in range(len(labels))]
    tf.reshape(b_o, [batch_size, 1])
    s_o = tf.add(u_o, b_o) # differences between biases and b_o?

    k = samples.size

    b_x = tf.gather(biases, samples)
    tf.reshape(b_x, [k, 1])
    s_x = tf.add(u_x, b_x)


    # A = log(sigma(s_o) - log(num_sampled * P(w_o)))
    # B = sum(log(1 - C))
    # C = sigma(s_x) - log(num_sampled * P(w_x))
    # sigma_matrix(x) =
