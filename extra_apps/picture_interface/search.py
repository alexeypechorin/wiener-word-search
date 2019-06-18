# encoding=utf-8

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

from io import BytesIO
import base64

from picture_word.settings import BASE_DIR
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

def build_phoc_descriptor(words, phoc_unigrams, unigram_levels,  #pylint: disable=too-many-arguments, too-many-branches, too-many-locals
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
    if on_unknown_unigram not in ['error', 'warn','nothing']:
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

        n = len(word) #pylint: disable=invalid-name
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
                        feat_vec_index = sum([l for l in unigram_levels if l < level]) * len(phoc_unigrams) + region * len(phoc_unigrams) + char_index
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

def run_query(queries, candidate_phocs, unigrams, unigram_levels=[1,2,4,8,16]):
    query_phocs = build_phoc_descriptor(queries, unigram_levels=unigram_levels, phoc_unigrams=unigrams)

    dist = cdist(query_phocs, candidate_phocs, 'cosine')
    sorted_results = np.argsort(dist, axis=1)

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

def getImg(str):
    print('loading vocabulary, words...')
    with open(os.path.join(BASE_DIR,'extra_apps','picture_interface','data', 'vocabulary.json'), 'r') as f:
        vocabulary = json.load(f)

    with open(os.path.join(BASE_DIR,'extra_apps','picture_interface','data', 'words.json'), 'r') as f:
        words = json.load(f)

    with open(os.path.join(BASE_DIR,'extra_apps','picture_interface','data', 'vocab_strings.json'), 'r') as f:
        vocab_strings = json.load(f)

    unigrams = []

    # load unigrams if they exist, else create them.
    if os.path.exists(os.path.join(BASE_DIR,'extra_apps','picture_interface','data', 'unigrams.json')):
        with open(os.path.join(BASE_DIR,'extra_apps','picture_interface','data', 'unigrams.json'), 'r') as f:
            unigrams = json.load(f)
    else:
        # create unigrams for all vocabulary
        unigrams = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        unigrams += [chr(i) for i in range(ord('0'), ord('9') + 1)]
        unigrams += [chr(i) for i in range(ord('À'), ord('ü'))]
        unigrams = sorted(unigrams)

        # save unigrams
        with open(os.path.join(BASE_DIR,'extra_apps','picture_interface','data', 'unigrams.json'), 'w') as f:
            json.dump(unigrams, f)

    candidates = []

    print('building candidates...')

    if os.path.exists(os.path.join(BASE_DIR,'extra_apps','picture_interface','data', 'candidates.npy')):
        print('loading candidates')
        candidates = np.load(os.path.join(BASE_DIR,'extra_apps','picture_interface','data', 'candidates.npy'))
    else:
        # build candidates and save them
        candidates = build_phoc_descriptor(vocab_strings, phoc_unigrams=unigrams, unigram_levels=[1,2,4,8,16])

        # save candidates
        np.save(os.path.join('data', 'candidates.npy'), candidates)

    queries = str.split()

    results = run_query(queries, candidates, unigrams)

    # get top 20 results for each query
    clean_results = show_clean_results(queries, results, vocab_strings, vocabulary, words, k=3)

    # just check that each one is properly showing the right thing
    # comment them out if you don't want to see them
    img_list=[]
    for query, result_objs in clean_results.items():
        print(query)

        # Group results by image
        result_objs_grouped = {}
        for result_obj in result_objs:
            if not result_obj['img_path'] in result_objs_grouped:
                result_objs_grouped[result_obj['img_path']] = []
            result_objs_grouped[result_obj['img_path']].append(result_obj)

        for result_obj_path, result_obj in result_objs_grouped.items():
            img = cv2.imread("extra_apps/picture_interface/{0}".format(result_obj_path))
            img_list.append((show_bboxes_with_text(img, result_obj), result_obj))

    image_src=[]
    for img, result_obj in img_list:
        # print(type(img))
        sio = BytesIO()
        img.save(sio, format='png')
        imgdata = base64.encodebytes(sio.getvalue()).decode()
        img_src = 'data:image/png;base64,' + imgdata
        image_src.append((img_src, result_obj[0]['img_path']))
    return image_src

def getImgSrc(str):
    img_list=getImg(str)
    return img_list


# if __name__ == '__main__':
#     img_list=getImgSrc('hat')