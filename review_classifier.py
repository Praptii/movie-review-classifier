import numpy as np
import math as ma
import re
from operator import itemgetter



def dataset_generator(positive_reviews, negative_reviews):
    positive_review_list = []
    negative_review_list= []
    positive_labels = None
    negative_labels = None

    count = 0
    with open(positive_reviews, encoding = 'utf-8') as fr:
        while True:
            line = fr.readline()

            positive_review_list.append(line)

            if not line:
                break
            count += 1

    count = 0
    with open(negative_reviews, encoding = 'utf-8') as fr:
        while True:
            line = fr.readline()

            negative_review_list.append(line)

            count += 1

            if not line:
                break

    positive_labels = [1] * len(positive_review_list)
    negative_labels = [0] * len(negative_review_list)

    return positive_review_list, positive_labels, negative_review_list, negative_labels

pos_data, pos_labels, neg_data, neg_labels = dataset_generator('train/allpos.txt','train/allneg.txt')

X_train, Y_train = pos_data + neg_data, pos_labels + neg_labels

pos_data, pos_labels, neg_data, neg_labels = dataset_generator('test/allpos.txt','test/allneg.txt')

X_test, Y_test = pos_data + neg_data, pos_labels + neg_labels


def line_preprocessor(string):
    stop_words = stopwords = ['the', 'of', 'and', 'is','to','in','a','from','by','that',
                              'with', 'this', 'as', 'an', 'are','its', 'at', 'for', 'it']

    string = string.lower()

    #Remove Numbers
    string = re.sub(r'[0-9]', '', string)

    # Remove punctuation
    string = re.sub(r'[^\w\s]','',string)

    words = string.split(' ')
    words = [x if x not in stop_words else ' ' for x in words]

    string  = ' '.join(words)

    return string

def word_freq_generator(articles):
    word_count = dict()

    for article in articles:
        article = line_preprocessor(article)
        article_words = article.split(' ')

        for word in article_words:
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1

    return word_count


def vocab_frequency_generator(total_dataset):

    word_freq = {}

    for sample in total_dataset:
        line = preprocess_string(sample[0])

        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    return word_freq


def vocab_generator(word_freq, freq_threshold):
    thresholded_word_freq = {k:v for (k,v) in word_freq.items() if v > freq_threshold}

    words = list(thresholded_word_freq.keys())

    vocab_dict = {x : words[x] for x in range(len(words))}

    # Writing the vocab to a text file
    with open("vocab.txt", 'w', encoding='utf-8') as f:
        for key, value in vocab_dict.items():
            f.write('%s:%s\n' % (key, value))

    return vocab_dict



def filter_words(word_count, threshold_k):

    # https://www.geeksforgeeks.org/python-n-largest-values-in-dictionary/
    top_k = dict(sorted(word_count.items(), key = itemgetter(1), reverse = True)[:threshold_k])

    return dict(enumerate(top_k))

def vocab_dataset_builder(positive_reviews_file, negative_reviews_file, num_words):
    pos_data, pos_labels, neg_data, neg_labels = dataset_generator(positive_reviews_file,negative_reviews_file)

    word_count = word_freq_generator(pos_data + neg_data)

    filtered = filter_words(word_count, 3000)

    filtered[len(filtered)] = 'unk'

    #https://www.geeksforgeeks.org/write-a-dictionary-to-a-file-in-python/
    with open("vocab.txt", 'w') as f:
        for key, value in filtered.items():
            f.write('%s:%s\n' % (key, value))

    return filtered

vocab_hash = vocab_dataset_builder('train/allpos.txt', 'train/allneg.txt', 30000)
rev_vocab_hash = {value : key for (key, value) in vocab_hash.items()}



def b_o_w_builder(sentence, vocab_hash):
    word_freq = dict.fromkeys(vocab_hash, 0)

    #https://www.kite.com/python/answers/how-to-reverse-a-dictionary-in-python
    rev_vocab_hash = {value : key for (key, value) in vocab_hash.items()}


    pped_line = line_preprocessor(sentence)

    words = pped_line.split(' ')

    for word in words:

        if word in vocab_hash.values():


            word_freq[rev_vocab_hash[word]] += 1

    return np.array( list(word_freq.values()))


def calculate_accuracy(actual_labels, predicted_labels):
    total = len(actual_labels)
    correct_ones = 0
    for a,p in zip(actual_labels, predicted_labels):
        if a == p:
            correct_ones = correct_ones + 1
    return correct_ones/total

def test_model(model,X_train, Y_train, X_test, Y_test, vocab_hash, rev_vocab_hash):


    if model == 'NB':

        prediction_train = []
        prediction_test = []

        for sample in X_train:
            prediction_train.append(naive_bayes_predict(sample, rev_vocab_hash))

        for sample in X_test:
            prediction_test.append(naive_bayes_predict(sample, rev_vocab_hash))

        print('**************** NAIVE BAYES ****************')
        print('Your train accuracy is :: ',calculate_accuracy(Y_train, prediction_train) )
        print('Your test accuracy is :: ',calculate_accuracy(Y_test, prediction_test) )


    if model == 'Perceptron':

        prediction_train = []
        prediction_test = []

        for sample in X_train:
            prediction_train.append(perceptron_predict(sample, vocab_hash))

        for sample in X_test:
            prediction_test.append(perceptron_predict(sample, vocab_hash))

        print('**************** Perceptron ****************')
        print('Your train accuracy is :: ', calculate_accuracy(Y_train, prediction_train) )
        print('Your test accuracy is :: ', calculate_accuracy(Y_test, prediction_test) )



### Naive Bayes

def naive_bayes_trainer(pos_samples, neg_samples, vocab_hash):
    positive_weights = np.zeros(len(vocab_hash))
    negative_weights =  np.zeros(len(vocab_hash))

    N_pos = len(pos_samples)
    N_neg = len(neg_samples)
    N_all = N_pos + N_neg

    count = 0

    # Since the dataset size for both positive samples and negative samples is the same
    for sample1, sample2 in zip(pos_samples, neg_samples):
        positive_weights += b_o_w_builder(sample1, vocab_hash)
        negative_weights += b_o_w_builder(sample2, vocab_hash)
        count += 1

    #Smoothing
    positive_weights += 1
    negative_weights += 1

    cond_probs_pos = positive_weights / positive_weights.sum()
    cond_probs_neg = negative_weights / negative_weights.sum()

    prior_pos = N_pos / N_all
    prior_neg = N_neg / N_all

    return positive_weights, prior_pos, negative_weights, prior_neg

pos_probs, pos_prior, neg_probs, neg_prior = naive_bayes_trainer(pos_data, neg_data, vocab_hash)



def naive_bayes_predict(review, reverse_vocab_hash):

    review = line_preprocessor(review)

    words = review.split(' ')
    idxes = []


    for word in words:
        if word in reverse_vocab_hash:
            idxes.append(reverse_vocab_hash[word])
        else:
            idxes.append(reverse_vocab_hash['unk'])

    pos_log_likelihood = np.log(pos_probs)
    neg_log_likelihood = np.log(neg_probs)

    sum_score_pos= 0
    sum_score_neg = 0
    for ids in idxes:
        sum_score_pos += pos_log_likelihood[ids]
        sum_score_neg += neg_log_likelihood[ids]

    sum_score_pos += np.log(pos_prior)
    sum_score_neg += np.log(neg_prior)

    if sum_score_pos > sum_score_neg:
        return 1
    else:
        return 0

### Perceptron

import random
def perceptron_train(X_t, Y_t, X_te, Y_te):
    learning_rate = 0.1
    weights = np.random.normal(0, 5,len(vocab_hash))

    # Modification 2: Addition of Bias
    bias = -2

    # Modification 1: Shuffling Dataset before training
    c1 = list(zip(X_t, Y_t))
    random.shuffle(c1)
    X_t, Y_t = zip(*c1)

    c2 = list(zip(X_te, Y_te))
    random.shuffle(c2)
    X_te, Y_te = zip(*c2)

    # 10 epochs
    for i in range(10):
        y_pred = []

        counter = 0

        for sample,label in zip(X_t, Y_t):
            data = b_o_w_builder(sample, vocab_hash)

            score = np.sum(data * weights) + bias

            result = None

            if score > 0:
                result = 1
            else:
                result = 0

            if result != label:
                weights = weights + (data) * (label - result) * learning_rate
                bias = bias + (label-result) * learning_rate

            y_pred.append(result)
            counter += 1

        # print('Training Accuracy ::: ', calculate_accuracy(Y_t, y_pred))

    return weights, bias


perceptron_weights, bias = perceptron_train(X_train, Y_train,  X_test, Y_test)


def perceptron_predict(review, vocab_hash):
    score = np.sum(b_o_w_builder(review, vocab_hash) * perceptron_weights) + bias
    print(score)
    if score > 0:
        return 1
    else:
        return 0

test_model('NB',X_train, Y_train, X_test, Y_test, vocab_hash, rev_vocab_hash)
test_model('Perceptron',X_train, Y_train, X_test, Y_test, vocab_hash, rev_vocab_hash)
