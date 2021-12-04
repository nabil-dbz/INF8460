# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:14:36 2021
"""

import numpy as np
import tensorflow_probability as tfp

def compute_transition_and_initial_matrix(train_data, tag_index) :
    
    initial_matrix = np.zeros(len(tag_index))
    transition_matrix = np.zeros((len(tag_index),len(tag_index)))
    sum_matrix = np.zeros(len(tag_index))
    i=0

    while i < len(train_data)-1 :
        current_doc_id = train_data.loc[i]["DocID"]
        beginning_tag_index = tag_index[train_data.loc[i]["Tag"]]
        initial_matrix[beginning_tag_index] += 1/len(train_data)

        while train_data.loc[i]["DocID"] == current_doc_id and i < len(train_data)-1 :
            current_tag_index = tag_index[train_data.loc[i]["Tag"]]
            next_tag_index = tag_index[train_data.loc[i+1]["Tag"]]
            transition_matrix[current_tag_index][next_tag_index] += 1
            sum_matrix[current_tag_index] += 1
            i+=1
        sum_matrix[next_tag_index] += 1

    for index_a in range(len(sum_matrix)) :
        for index_b in range(len(transition_matrix[index_a])) :
            transition_matrix[index_a][index_b] = transition_matrix[index_a][index_b]/sum_matrix[index_a]

    return initial_matrix, transition_matrix


def compute_emission_matrix(train_data, types_index, tag_index) :
    
    emission_matrix = np.zeros((len(tag_index),len(types_index)))
    sum_matrix = np.zeros(len(tag_index))
    
    for i in range(len(train_data)) :
        current_tag_index = tag_index[train_data.loc[i]["Tag"]]
        current_type_index = types_index[train_data.loc[i]["Token"]]
        emission_matrix[current_tag_index][current_type_index] += 1
        sum_matrix[current_tag_index] += 1
        
    for index_a in range(len(sum_matrix)) :
        for index_b in range(len(emission_matrix[index_a])) :
            emission_matrix[index_a][index_b] = emission_matrix[index_a][index_b]/sum_matrix[index_a]
            
    return emission_matrix

def create_distributions(initial_matrix, transition_matrix, emission_matrix) :
    tfd = tfp.distributions
    initial_distribution = tfd.Categorical(probs=initial_matrix)
    transition_distribution = tfd.Categorical(probs=transition_matrix)
    observation_distribution = tfd.Categorical(probs=emission_matrix)
    return tfd, initial_distribution, transition_distribution, observation_distribution


def get_output_hmm_model(tfd, initial_distribution, transition_distribution, observation_distribution, sentences, reverse_tag_index) :
    output = []
    for sentence in sentences :
        model = tfd.HiddenMarkovModel(initial_distribution=initial_distribution,
                                  transition_distribution=transition_distribution,
                                  observation_distribution=observation_distribution,
                                  num_steps = len(sentence))
        tag_sequence = model.posterior_mode(observations=sentence)
        reversed_tag_index = [reverse_tag_index[index] for index in tag_sequence.numpy()]
        output = output + reversed_tag_index

    return output

