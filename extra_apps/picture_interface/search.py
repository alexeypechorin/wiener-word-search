# encoding=utf-8
import time

from io import BytesIO
from picture_word.settings import BASE_DIR
import base64
import os
import numpy as np
from tqdm import tqdm
import json
import string
from scipy.spatial.distance import cdist, pdist, squareform
from PIL import Image
import cv2
import sys
import importlib

importlib.reload(sys)


def correct():
    pass


def show_bboxes_with_text(img, boxes, sym_spell=None):
    candidate_corrections = []
    for i, box in enumerate(boxes):
        bbox = box['bbox']

        if sym_spell is not None:
            candidate_correction = correct(box['word'], sym_spell)
        else:
            candidate_correction = ''
        # box is in absolute coordinates
        t = '[' + box['word'] + ']'
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), thickness=1)
        fontFace = 0
        fontScale = .2 * (img.shape[0] / 720.0)
        thickness = 1
        fg = (0, 120, 0)
        textSize, baseline = cv2.getTextSize(t, fontFace, fontScale, thickness)
        cv2.putText(img, t, (bbox[0], bbox[1]),
                    fontFace, fontScale, fg, thickness)

        candidate_corrections += [candidate_correction]

    img = Image.fromarray(img)
    # img.show()

    return img


def build_phoc_descriptor(words, phoc_unigrams, unigram_levels,
                          # pylint: disable=too-many-arguments, too-many-branches, too-many-locals
                          bigram_levels=None, phoc_bigrams=None,
                          split_character=None, on_unknown_unigram='nothing',
                          phoc_type='phoc'):
    '''
    Calculate Pyramidal Histogram of Characters (PHOC) descriptor (see Almazan 2014).

    Args:
        word (str): word to calculate descriptor for
        phoc_unigrams (str): string of all unigrams to use in the PHOC
        unigram_levels (list of int): the levels to use in the PHOC
        split_character (str): special character to split the word strings into characters
        on_unknown_unigram (str): What to do if a unigram appearing in a word
            is not among the supplied phoc_unigrams. Possible: 'warn', 'error', 'nothing'
        phoc_type (str): the type of the PHOC to be build. The default is the
            binary PHOC (standard version from Almazan 2014).
            Possible: phoc, spoc
    Returns:
        the PHOC for the given word
    '''
    # prepare output matrix
    if on_unknown_unigram not in ['error', 'warn', 'nothing']:
        raise ValueError('I don\'t know the on_unknown_unigram parameter \'%s\'' % on_unknown_unigram)
    phoc_size = len(phoc_unigrams) * np.sum(unigram_levels)
    if phoc_bigrams is not None:
        phoc_size += len(phoc_bigrams) * np.sum(bigram_levels)
    phocs = np.zeros((len(words), phoc_size))
    # prepare some lambda functions
    occupancy = lambda k, n: [float(k) / n, float(k + 1) / n]
    overlap = lambda a, b: [max(a[0], b[0]), min(a[1], b[1])]
    size = lambda region: region[1] - region[0]

    # map from character to alphabet position
    char_indices = {d: i for i, d in enumerate(phoc_unigrams)}

    # iterate through all the words
    for word_index, word in enumerate(tqdm(words)):
        if split_character is not None:
            word = word.split(split_character)

        n = len(word)  # pylint: disable=invalid-name
        for index, char in enumerate(word):
            char_occ = occupancy(index, n)
            if char not in char_indices:
                if on_unknown_unigram == 'warn':
                    logger.warn('The unigram \'%s\' is unknown, skipping this character', char)
                    continue
                elif on_unknown_unigram == 'error':
                    logger.fatal('The unigram \'%s\' is unknown', char)
                    raise ValueError()
                else:
                    continue
            char_index = char_indices[char]
            for level in unigram_levels:
                for region in range(level):
                    region_occ = occupancy(region, level)
                    if size(overlap(char_occ, region_occ)) / size(char_occ) >= 0.5:
                        feat_vec_index = sum([l for l in unigram_levels if l < level]) * len(
                            phoc_unigrams) + region * len(phoc_unigrams) + char_index
                        if phoc_type == 'phoc':
                            phocs[word_index, feat_vec_index] = 1
                        elif phoc_type == 'spoc':
                            phocs[word_index, feat_vec_index] += 1
                        else:
                            raise ValueError('The phoc_type \'%s\' is unknown' % phoc_type)
        # add bigrams
        if phoc_bigrams is not None:
            ngram_features = np.zeros(len(phoc_bigrams) * np.sum(bigram_levels))
            ngram_occupancy = lambda k, n: [float(k) / n, float(k + 2) / n]
            for i in range(n - 1):
                ngram = word[i:i + 2]
                if phoc_bigrams.get(ngram, 0) == 0:
                    continue
                occ = ngram_occupancy(i, n)
                for level in bigram_levels:
                    for region in range(level):
                        region_occ = occupancy(region, level)
                        overlap_size = size(overlap(occ, region_occ)) / size(occ)
                        if overlap_size >= 0.5:
                            if phoc_type == 'phoc':
                                ngram_features[region * len(phoc_bigrams) + phoc_bigrams[ngram]] = 1
                            elif phoc_type == 'spoc':
                                ngram_features[region * len(phoc_bigrams) + phoc_bigrams[ngram]] += 1
                            else:
                                raise ValueError('The phoc_type \'%s\' is unknown' % phoc_type)
            phocs[word_index, -ngram_features.shape[0]:] = ngram_features
    return phocs


def run_query(queries, candidates, Wx, mean_x, hub_matrix, unigrams, unigram_levels=[1, 2, 4, 8], num_nn=20):
    # assumes that candidates are already projected and normalized
    tic = time.process_time()
    query_phocs = build_phoc_descriptor(queries, unigram_levels=unigram_levels, phoc_unigrams=unigrams)
    toc = time.process_time()

    print('building phoc descriptor time:', toc - tic)

    tic = time.process_time()
    # project and normalize
    projected_query = np.matmul(query_phocs - np.transpose(mean_x).reshape(1, -1), Wx)
    query_norms = np.linalg.norm(projected_query, axis=1)
    query_norms = np.reshape(query_norms, (-1, 1))
    projected_query = projected_query / query_norms
    toc = time.process_time()

    print('projecting time:', toc - tic)

    tic = time.process_time()
    # find distance from projected query to candidates using cosine distance
    dist = 1 - np.matmul(projected_query, np.transpose(candidates))
    toc = time.process_time()

    print('main distance time:', toc - tic)

    tic = time.process_time()
    # find avg distance to nearest neighbors for query
    sorted_distances = np.sort(dist, axis=-1)
    trunc_distances = sorted_distances[:, :num_nn]
    avg_nn_dist = np.sum(trunc_distances, axis=1) / num_nn
    toc = time.process_time()

    print('nearest neighbor time:', toc - tic)

    tic = time.process_time()
    dist = 2 * dist - avg_nn_dist - np.transpose(hub_matrix)
    toc = time.process_time()

    print('final distance calculation:', toc - tic)

    tic = time.process_time()
    # sort the final results
    sorted_results = np.argsort(dist, axis=1)
    toc = time.process_time()

    print('sorting time:', toc - tic)

    return sorted_results


def show_clean_results(queries, results, vocab_strings, vocabulary, words, k=20):
    """
    prints out clean table of results
    :param results:
    :param vocabulary:
    :param words:
    :return: a list containing top k results (more than k if multiple occurrence of some word string)
    """

    top_k = {}

    for row in range(results.shape[0]):
        top_k[queries[row]] = []
        print(queries[row] + ':')
        print('------------------')
        for res_idx in range(k):
            result_str = vocab_strings[results[row, res_idx]]
            print(res_idx, result_str)

            indices = vocabulary[result_str]

            top_k[queries[row]] += [words[idx] for idx in indices]

    return top_k


def load_vocabulary_data(unigrams_file_path, vocab_strings_file_path, vocabulary_file_path, words_file_path):
    print('Loading vocabulary, words...')
    tic_vocab = time.process_time()
    with open(vocabulary_file_path, 'r') as f:
        vocabulary = json.load(f)
    with open(words_file_path, 'r') as f:
        words = json.load(f)
    with open(vocab_strings_file_path, 'r') as f:
        vocab_strings = json.load(f)
    with open(unigrams_file_path, 'r') as f:
        unigrams = json.load(f)
    toc_vocab = time.process_time()
    print(toc_vocab - tic_vocab, 'seconds...')
    return unigrams, vocab_strings, vocabulary, words


def load_model_data(Wx_file_path, candidates_file_path, hub_matrix_file_path, mean_x_file_path):
    print('Loading all candidates')
    tic_cands = time.process_time()
    candidates = np.load(candidates_file_path)
    toc_cands = time.process_time()
    print(toc_cands - tic_cands, 'seconds...')
    tic_hub = time.process_time()
    print('Loading Wx, mean_x, and hub_matrix...')
    Wx = np.load(Wx_file_path)
    mean_x = np.load(mean_x_file_path)
    hub_matrix = np.load(hub_matrix_file_path)
    toc_hub = time.process_time()
    print(toc_hub - tic_hub, 'seconds...')
    return Wx, candidates, hub_matrix, mean_x


def get_img(queries_str, model_data):
    model_data_folder = 'extra_apps/picture_interface/model_data_deu'
    candidates_file_path = os.path.join(model_data_folder, 'candidates_all.npy')
    Wx_file_path = os.path.join(model_data_folder, 'Wx.npy')
    mean_x_file_path = os.path.join(model_data_folder, 'mean_x.npy')
    hub_matrix_file_path = os.path.join(model_data_folder, 'hub.npy')
    vocabulary_file_path = os.path.join(model_data_folder, 'vocabulary.json')
    words_file_path = os.path.join(model_data_folder, 'words.json')
    vocab_strings_file_path = os.path.join(model_data_folder, 'vocab_strings.json')
    unigrams_file_path = os.path.join(model_data_folder, 'unigrams.json')

    if not model_data:
        Wx, candidates, hub_matrix, mean_x = load_model_data(Wx_file_path, candidates_file_path,
                                                             hub_matrix_file_path, mean_x_file_path)
        unigrams, vocab_strings, vocabulary, words = load_vocabulary_data(unigrams_file_path, vocab_strings_file_path,
                                                                          vocabulary_file_path, words_file_path)

        model_data = {'Wx': Wx,
                        'candidates': candidates,
                        'hub_matrix': hub_matrix,
                        'mean_x': mean_x,
                        'unigrams': unigrams,
                        'vocab_strings': vocab_strings,
                        'vocabulary': vocabulary,
                        'words': words
                      }
    else:
        Wx, candidates, hub_matrix, mean_x, unigrams, vocab_strings, vocabulary, words = \
        model_data['Wx'], model_data['candidates'], model_data['hub_matrix'], model_data['mean_x'], model_data[
            'unigrams'], model_data['vocab_strings'], model_data['vocabulary'], model_data['words']

    # get top 20 results for each query
    queries = queries_str.split()
    tic = time.process_time()
    results = run_query(queries, candidates, Wx, mean_x, hub_matrix, unigrams)
    toc = time.process_time()
    print(toc - tic, 'seconds...')

    clean_results = show_clean_results(queries, results, vocab_strings, vocabulary, words, k=3)

    # just check that each one is properly showing the right thing
    # comment them out if you don't want to see them
    img_list = []
    for query, result_objs in clean_results.items():
        print(query)

        # Group results by image
        result_objs_grouped = {}
        for result_obj in result_objs:
            if not result_obj['img_path'] in result_objs_grouped:
                result_objs_grouped[result_obj['img_path']] = []
            result_objs_grouped[result_obj['img_path']].append(result_obj)

        for result_obj_path, result_obj in result_objs_grouped.items():
            img = cv2.imread(result_obj_path)
            img_list.append((show_bboxes_with_text(img, result_obj), result_obj))

    image_src = []
    for img, result_obj in img_list:
        # print(type(img))
        sio = BytesIO()
        img.save(sio, format='png')
        imgdata = base64.encodebytes(sio.getvalue()).decode()
        img_src = 'data:image/png;base64,' + imgdata
        image_src.append((img_src, result_obj[0]['img_path']))
    return image_src, model_data


def get_img_src(query_str, model_data):
    img_list, model_data = get_img(query_str, model_data)
    return img_list, model_data

# if __name__ == '__main__':
#     img_list=getImgSrc('hat')
