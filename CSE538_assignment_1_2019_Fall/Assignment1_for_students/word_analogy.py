import os
import pickle
import numpy as np
import scipy as sc
import sklearn as sk
from sklearn.metrics.pairwise import cosine_similarity

model_path = './models/'
loss_model = 'cross_entropy'
# loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""
# examples || map
filename = './word_analogy_dev.txt'

with open(filename) as f:
    line = f.readline()

    output = []

    while line:
        # Strip trailing newline
        line = line.strip('\n')

        # split into examples and choices
        big_list_components = line.split("||")
        ex_comp = big_list_components[0]
        choices_comp = big_list_components[1]

        # Split them by comma
        examples = ex_comp.split(",")
        choices = choices_comp.split(",")
        # print(examples)
        # print(choices)

        # Put in list, removing "" in Process
        example_diff_list = [] # Embedding differences - examples
        choice_diff_list = [] # Embedding differences - choices

        for ex in examples:
            ex0 = ex[1:-1].split(":") # "ex1:ex2"
            ex1 = ex0[0]
            ex2 = ex0[1]

            e_word_id_1 = dictionary[ex1]
            e_word_id_2 = dictionary[ex2]

            e_emb1 = embeddings[e_word_id_1]
            e_emb2 = embeddings[e_word_id_2]

            ex_diff = e_emb1 - e_emb2
            example_diff_list.append(ex_diff)

        for c in choices:
            c0 = c[1:-1].split(":")
            print(c0)
            c1 = c0[0]
            c2 = c0[1]
            # print(c1)
            # print(c2)

            c_word_id_1 = dictionary[c1]
            c_word_id_2 = dictionary[c2]

            c_emb1 = embeddings[c_word_id_1]
            c_emb2 = embeddings[c_word_id_2]

            c_diff = c_emb1 - c_emb2
            choice_diff_list.append(c_diff)

        cs = cosine_similarity(example_diff_list, choice_diff_list)
        # print(cs)
        # print(cs[0])

        avg_cs = []
        for i in range(0, len(cs[0])):
            sum_i = 0
            for j in range(0, len(cs)):
                sum_i += cs[j][i]
            avg_i = float(sum_i) / len(cs[0])
            avg_cs.append(avg_i)

        print(avg_cs)

        least_similar_pair = choices[avg_cs.index(min(avg_cs))]
        most_similar_pair = choices[avg_cs.index(max(avg_cs))]

        print(least_similar_pair)
        print(most_similar_pair)

        choice_str = ""
        for c in choices:
            print(c)
            choice_str += c + " "

        output.append(choice_str + " " + least_similar_pair + " " + most_similar_pair)

        line = f.readline()

if loss_model == 'cross_entropy':
    output_file = "test_dev_ce.txt"
elif loss_model == "nce":
    output_file = "test_dev_nce.txt"

f_o = open(output_file, 'w')
for l2 in output:
    f_o.write(l2 + "\n")

f_o.close()
