import datasets
import tqdm
import spacy
import numpy as np
from Config import *

def save_to_file(file_name, contents):
    fh = open(file_name, 'w', encoding="utf-8")
    for content in contents:
        for one in content:
            fh.write(f"{content.index(one)}: \n")
            for k, v in one.items():
                fh.write(f"{k} : {v}")
            fh.write(f"\n")
    fh.close()

def data_load(path):
    train = datasets.load_from_disk(path + "/train")
    valid = datasets.load_from_disk(path + "/valid")
    return train, valid


def data2np(data):
    words = []
    labels = []
    for content, label in zip(data["content"], data["personality"]):
        word = content.split("|||")
        word.remove("")
        words.append(word)
        labels.append(label)
    return words, labels


def document_creator(words, is_tqdm):
    document = ""
    words = tqdm.tqdm(words, position=0, leave=False, desc="整合长文本") if is_tqdm else words
    
    for word in words:
        document = "".join([document, " ".join(word)])
    return document


def word_counter(nlp, document, is_tqdm):
    document = nlp(document)
    
    word_count = {}
    document = tqdm.tqdm(document, position=1, leave=False, desc="词数统计") if is_tqdm else document
    
    for token in document:
        if token.text in word_count.keys():
            word_count[token.text] += 1
        else:
            word_count[token.text] = 1

    return word_count


def classifier_data(train, number):
    classification = [[] for i in range(8)]
    true_labels = ["I", "S", "T", "J"]
    
    if number > len(train[0]):
        number = -1
    for words, label in zip(train[0][0 : number], train[1][0 : number]):
        for i in range(4):
            if label[i] == true_labels[i]:
                classification[i * 2].append(words)
            else:
                classification[i * 2 + 1].append(words)
                
    return classification


def count_one(nlp, classification, is_tqdm):
    classification_count_every = [[] for i in range(8)]
    times = tqdm.tqdm(range(8), position=0, leave=False, desc="各种类别词数统计") if is_tqdm else range(8)
    for utype in times:
        for one in classification[utype]:
            document = " ".join(one)
            classification_count_every[utype].append(word_counter(nlp, document, is_tqdm))
    return classification_count_every


def count_four(nlp, classification, is_tqdm):
    counts = []
    for i in range(8):
        document = document_creator(classification[i], is_tqdm)
        pos = 0
        word_count = {}
        while pos < len(document):
            word_count.update(word_counter(nlp, document[pos : pos + 999999], is_tqdm))
            pos += 999999
        word_count.update(word_counter(nlp, document[pos : -1], is_tqdm))
        counts.append(word_count)
    return counts


def remove_special_word(count):
    total_num = len(count)
    min_limited = total_num * 0.001
    del_list = []
    for key, val in count.items():
        if val < min_limited:
            del_list.append(key)
    
    for key in del_list:
        count.pop(key)
    return count


def normalize(classification_count_every, valid_words):
    nml_count = [[] for i in range(8)]
    for utype in range(8):
        for one in classification_count_every[utype]:
            vec = []
            for word in valid_words[utype // 2]:
                if word in one.keys():
                    vec.append(one[word])
                else:
                    vec.append(0)
            nml_count[utype].append(vec)
    return nml_count


def get_feature_and_labels(nml_count):
    features = []
    labels = []
    for i in range(4):
        features.append(np.append(np.array(nml_count[2 * i]), np.array(nml_count[2 * i + 1]), axis=0))
        labels.append(np.append(np.array([1 for i in range(len(nml_count[2 * i]))]), np.array([0 for i in range(len(nml_count[2 * i + 1]))])))
    
    return features, labels

def get_valid_keys(I_count_t, E_count_t, S_count_t, N_count_t, T_count_t, F_count_t, J_count_t, P_count_t, is_flit):
    if is_flit:
        I_count_t = remove_special_word(I_count_t)
        E_count_t = remove_special_word(E_count_t)
        S_count_t = remove_special_word(S_count_t)
        N_count_t = remove_special_word(N_count_t)
        T_count_t = remove_special_word(T_count_t)
        F_count_t = remove_special_word(F_count_t)
        J_count_t = remove_special_word(J_count_t)
        P_count_t = remove_special_word(P_count_t)
    
    words_IE = list(set(list(I_count_t.keys())+list(E_count_t.keys())))
    words_SN = list(set(list(S_count_t.keys())+list(N_count_t.keys())))
    words_TF = list(set(list(T_count_t.keys())+list(F_count_t.keys())))
    words_JP = list(set(list(J_count_t.keys())+list(P_count_t.keys())))
    
    return words_IE, words_SN, words_TF, words_JP


def getData(path, is_flit, number, is_tqdm):
    train, valid = data_load(path)
    data_train = data2np(train)
    data_valid = data2np(valid)
    
    nlp = spacy.load("en_core_web_sm", exclude=["tok2vec", "tagger", "senter", "attribute_ruler", "lemmatizer"])
    
    classification_train = classifier_data(data_train, number)
    classification_valid = classifier_data(data_valid, number)
    
    classification_train_count_every = count_one(nlp, classification_train, is_tqdm)
    classification_valid_count_every = count_one(nlp, classification_valid, is_tqdm)
    
    # save_to_file("classification_train_count_every.txt", classification_train_count_every)
    # save_to_file("classification_valid_count_every.txt", classification_valid_count_every)
    
    I_count_t, E_count_t, S_count_t, N_count_t, T_count_t, F_count_t, J_count_t, P_count_t = count_four(nlp, classification_train, is_tqdm)
    I_count_v, E_count_v, S_count_v, N_count_v, T_count_v, F_count_v, J_count_v, P_count_v = count_four(nlp, classification_valid, is_tqdm)
    
    words_IE, words_SN, words_TF, words_JP = get_valid_keys(I_count_t, E_count_t, S_count_t, N_count_t, T_count_t, F_count_t, J_count_t, P_count_t, is_flit)
    
    nml_count_train = normalize(classification_train_count_every, [words_IE, words_SN, words_TF, words_JP])
    nml_count_valid = normalize(classification_valid_count_every, [words_IE, words_SN, words_TF, words_JP])
    
    features_train, labels_train = get_feature_and_labels(nml_count_train)
    features_valid, labels_valid = get_feature_and_labels(nml_count_valid)
    
    return features_train, labels_train, features_valid, labels_valid
    