import cv2, os, random, sys
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from scipy.spatial.distance import cdist

# Dataset info extraction
def create_image_info_list(dataset_path):
    image_info_list = []
    for city_folder in os.listdir(dataset_path):
        city_folder_path = Path(dataset_path) / city_folder
        if not city_folder_path.is_dir():
            continue
        for image_file in os.listdir(city_folder_path):
            if not image_file.endswith(".jpg"):
                continue
            parts = image_file.split('_')
            if len(parts) > 8:
                parts = parts[:7] + ['_'.join(parts[7:])]
            (city, place_id, year, month, bearing, latitude,
                longitude, panoid) = parts
            panoid = panoid.split('.')[0]
            image_info = {
                'city': city,
                'placeID': place_id,
                'year': year,
                'month': month,
                'bearing': bearing,
                'latitude': latitude,
                'longitude': longitude,
                'panoid': panoid,
                'path': str(city_folder_path / image_file)
            }
            image_info_list.append(image_info)
    return image_info_list

# SIFT feature extraction
def extract_sift_features(image_paths):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, descriptors = sift.detectAndCompute(gray, None)
        descriptors_list.append(descriptors)
    return descriptors_list

# Vocabulary building
def build_vocabulary(descriptors_list, k):
    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto', tol=1)
    kmeans.fit(all_descriptors)
    vocabulary = kmeans.cluster_centers_
    return vocabulary

# Feature encoding
def build_features(vocabulary, descriptors_list):
    features = []
    for descriptors in descriptors_list:
        distances = cdist(descriptors, vocabulary, 'euclidean')
        closest_clusters = np.argmin(distances, axis=1)
        histogram = np.bincount(closest_clusters, minlength=len(vocabulary))
        features.append(histogram)
    return features

# Classifier training
def train_classifier(features, labels):
    classifier = SVC(kernel='linear')
    classifier.fit(features, labels)
    return classifier

random.seed(19260817)
image_path = sys.argv[1]
print('Read dataset list')
image_list = create_image_info_list(image_path)
print('Split training / test set')
test_list = random.sample(image_list, len(image_list) // 10)
image_list = [i for i in image_list if i not in test_list]
image_paths = [i['path'] for i in image_list]
print('Training size: {}; Test size: {}'.format(
    len(image_list),
    len(test_list))
)
print('Extract SIFT')
descriptors_list = extract_sift_features(image_paths)
sift_size = 0
for i in descriptors_list:
    sift_size += len(i)
print('SIFT size: {}'.format(sift_size))
print('Build vocabulary')
vocabulary = build_vocabulary(
    descriptors_list,
    k=790
)
print('Build features')
features = build_features(vocabulary, descriptors_list)
# print(features)
print('Train classifier')
labels = ['{}_{}'.format(i['city'], i['placeID']) for i in image_list]
classifier = train_classifier(features, labels)
print('Training completed')

test_paths = [i['path'] for i in test_list]
print('Extract test set SIFT')
test_descriptors_list = extract_sift_features(test_paths)
print('Build test set features')
test_features = build_features(vocabulary, test_descriptors_list)
# print(test_features)
print('Predict test set')
predictions = classifier.predict(test_features)

cnt = 0

for i in range(len(test_list)):
    label = '{}_{}'.format(test_list[i]['city'], test_list[i]['placeID'])
    print(label, predictions[i])
    if label == predictions[i]:
        cnt += 1

print('Accuracy: {:.2f}%'.format(100 * cnt / len(test_list)))
