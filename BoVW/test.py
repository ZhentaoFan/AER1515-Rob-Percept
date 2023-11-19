import os, sys, pickle
from constants import *
from functions import (
    load_image_info_list, extract_sift_features, build_features
)

def main():
    try:
        image_path = sys.argv[1]
        accuracy_test = os.path.isdir(image_path)
    except:
        print('Invalid argv! Expected: test_set_path / test_image')

    print('Load k-means model')

    with open(F_KMEANS, 'rb') as file:
        kmeans = pickle.load(file)

    print('Load classifier model')

    with open(F_CLASSIFIER, 'rb') as file:
        classifier = pickle.load(file)

    if not accuracy_test:
        print('Extract test image SIFT')
        test_descriptors = extract_sift_features([image_path])

        print('Build test image features')
        test_features = build_features(kmeans, test_descriptors)

        print('Prediction:', classifier.predict(test_features)[0])

        return

    print('Read test set list')

    test_list = load_image_info_list(os.path.join(image_path, 'test'))
    test_paths = [i[K_PATH] for i in test_list]

    print('Extract test set SIFT')

    test_descriptors = extract_sift_features(test_paths)

    print('Build test set features')

    test_features = build_features(kmeans, test_descriptors)

    print('Predict test set')

    predictions = classifier.predict(test_features)
    cnt = 0

    for i in range(len(test_list)):
        label = f'{test_list[i][K_CITY]}_{test_list[i][K_PLACEID]}'
        if label == predictions[i]:
            cnt += 1
        else:
            print(label, predictions[i])

    print('Accuracy: {:.2f}%'.format(100 * cnt / len(test_list)))

if __name__ == '__main__':
    main()
