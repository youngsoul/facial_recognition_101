from face_recognition.api import face_distance
import argparse
import pickle
import cv2
import face_recognition

def get_encoding_for_image(imagePath, detection_method):
    image = cv2.imread(imagePath)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x,y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    # we are assuming the the boxes of faces are the SAME FACE or SAME PERSON
    boxes = face_recognition.face_locations(rgb_image, model=detection_method)

    # compute the facial embedding for the face
    # creates a vector of 128 numbers representing the face
    encodings = face_recognition.face_encodings(rgb_image, boxes)

    # we are assuming that the picture only has a single face, and that it is safe to get the zero-th index
    return encodings[0]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--new-image", required=True,
                    help="path to the new image to encode and calculate distance")
    ap.add_argument("-e", "--encodings-file", required=True,
                    help="path to serialized pickle file of facial encodings.  If the file exists, new encodings will be added.  Otherwise the file will be created")
    ap.add_argument("-m", "--detection-method", type=str, required=False, default='hog',
                    help="face detection model to use: either 'hog' or 'cnn' ")
    args = vars(ap.parse_args())

    new_encoded_image = get_encoding_for_image(args['new_image'], args['detection_method'])

    results = pickle.loads(open(args['encodings_file'], "rb").read())

    encodings = results['encodings']
    names = results['names']
    distance_results = face_distance(encodings, new_encoded_image)
    results = list(zip(distance_results, names))
    sorted_list = sorted(results, key=lambda x: x[0])
    for x in sorted_list:
        print(x)

