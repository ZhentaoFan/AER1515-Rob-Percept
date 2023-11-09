from os.path import exists, isdir, basename, join, splitext
from glob import glob
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
import scipy.cluster.vq as vq
from pickle import dump, HIGHEST_PROTOCOL
import argparse

import cv2
import numpy as np
import os
import pickle

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
DATASETPATH = './dataset'
PRE_ALLOCATION_BUFFER = 1000  # for sift
HISTOGRAMS_FILE = 'trainingdata.svm'
K_THRESH = 100  # early stopping threshold for kmeans originally at 1e-5, increased for speedup
CODEBOOK_FILE = 'codebook.file'


def parse_arguments():
    parser = argparse.ArgumentParser(description='train a visual bag of words model')
    parser.add_argument('-d', help='path to the dataset', required=False, default=DATASETPATH)
    args = parser.parse_args()
    return args


def get_categories(datasetpath):
    cat_paths = [files
                 for files in glob(datasetpath + "/*")
                  if isdir(files)]
    cat_paths.sort()
    cats = [basename(cat_path) for cat_path in cat_paths]
    return cats


def get_imgfiles(path):
    all_files = []
    all_files.extend([join(path, basename(fname))
                    for fname in glob(path + "/*")
                    if splitext(fname)[-1].lower() in EXTENSIONS])
    return all_files


def extractSift(input_files):
    print("extracting SIFT features")
    all_features_dict = {}
    sift = cv2.SIFT_create()  # Create a SIFT object

    for i, fname in enumerate(input_files):
        features_fname = fname + '.sift'
        if not os.path.exists(features_fname):
            print("calculating SIFT features for", fname)
            image = cv2.imread(fname)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(gray_image, None)
            keypoints_serializable = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

            with open(features_fname, 'wb') as f:
                pickle.dump((keypoints_serializable, descriptors), f)
        else:
            print("gathering SIFT features for", fname)
            try:
                with open(features_fname, 'rb') as f:
                    keypoints_serializable, descriptors = pickle.load(f)
            except EOFError:
                print(f"Error reading file {features_fname}. It may be corrupt or incomplete.")
                continue  # Skip this file and move to the next

        # keypoints = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], _size=pt[1], _angle=pt[2], _response=pt[3], _octave=pt[4], _class_id=pt[5]) for pt in keypoints_serializable]

        print(descriptors.shape)
        all_features_dict[fname] = descriptors

    return all_features_dict


def dict2numpy(dict):
    nkeys = len(dict)
    array = zeros((nkeys * PRE_ALLOCATION_BUFFER, 128))
    pivot = 0
    for key in dict.keys():
        value = dict[key]
        nelements = value.shape[0]
        while pivot + nelements > array.shape[0]:
            padding = zeros_like(array)
            array = vstack((array, padding))
        array[pivot:pivot + nelements] = value
        pivot += nelements
    array = resize(array, (pivot, 128))
    return array


def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              density=True)
    return histogram_of_words


def writeHistogramsToFile(nwords, labels, fnames, all_word_histgrams, features_fname):
    data_rows = zeros(nwords + 1)  # +1 for the category label
    print(all_word_histgrams.keys())
    for fname in fnames:
        histogram = all_word_histgrams[fname]
        if (histogram.shape[0] != nwords):  # scipy deletes empty clusters
            nwords = histogram.shape[0]
            data_rows = zeros(nwords + 1)
            print('nclusters have been reduced to ' + str(nwords))
        data_row = hstack((labels[fname], histogram))
        data_rows = vstack((data_rows, data_row))
    data_rows = data_rows[1:]
    fmt = '%i '
    for i in range(nwords):
        fmt = fmt + str(i) + ':%f '
    savetxt(features_fname, data_rows, fmt)


if __name__ == '__main__':
    print("---------------------")
    print("## loading the images and extracting the sift features")
    args = parse_arguments()
    datasetpath = args.d
    cats = get_categories(datasetpath)
    ncats = len(cats)
    print("searching for folders at " + datasetpath)
    if ncats < 1:
        raise ValueError('Only ' + str(ncats) + ' categories found. Wrong path?')
    print("found following folders / categories:")
    print(cats)
    print("---------------------")
    all_files = []
    all_files_labels = {}
    all_features = {}
    cat_label = {}
    for cat, label in zip(cats, range(ncats)):
        cat_path = join(datasetpath, cat)
        cat_files = get_imgfiles(cat_path)
        cat_features = extractSift(cat_files)
        all_files = all_files + cat_files
        all_features.update(cat_features)
        cat_label[cat] = label
        for i in cat_files:
            all_files_labels[i] = label

    print("---------------------")
    print("## computing the visual words via k-means")
    all_features_array = dict2numpy(all_features)
    nfeatures = all_features_array.shape[0]
    nclusters = int(sqrt(nfeatures))
    print("ncluster: ", nclusters)
    codebook, distortion = vq.kmeans(all_features_array,
                                             nclusters,
                                             thresh=K_THRESH)

    with open(datasetpath + CODEBOOK_FILE, 'wb') as f:

        dump(codebook, f, protocol=HIGHEST_PROTOCOL)

    print("---------------------")
    print("## compute the visual words histograms for each image")
    all_word_histgrams = {}
    for imagefname in all_features:
        word_histgram = computeHistograms(codebook, all_features[imagefname])
        all_word_histgrams[imagefname] = word_histgram

    print("---------------------")
    print("## write the histograms to file to pass it to the svm")
    writeHistogramsToFile(nclusters,
                          all_files_labels,
                          all_files,
                          all_word_histgrams,
                          datasetpath + HISTOGRAMS_FILE)


    print("---------------------")
    print("## train svm")

    # Prepare your data for training
    X = []  # feature vectors (histograms)
    y = []  # labels

    for imagefname, word_histgram in all_word_histgrams.items():
        X.append(word_histgram)
        y.append(all_files_labels[imagefname])

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an SVM classifier
    clf = SVC(kernel='linear')  # You can choose different kernels such as 'rbf', 'poly', etc.

    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    # Save the trained model to a file
    model_file = 'svm_model.pkl'
    joblib.dump(clf, datasetpath + model_file)

    print("--------------------")
    print("## outputting results")
    print("model file: " + datasetpath + model_file)
    print("codebook file: " + datasetpath + CODEBOOK_FILE)
    print("category      ==>  label")
    for cat in cat_label:
        print('{0:13} ==> {1:6d}'.format(cat, cat_label[cat]))