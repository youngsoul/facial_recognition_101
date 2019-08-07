import time
from imutils import paths
import face_recognition
import pickle
import cv2
import os
import argparse
from pathlib import Path


def create_encodings(dataset, encodings_file, detection_method):
    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(dataset))

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []
    s = time.time()

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        name = imagePath.split(os.path.sep)[-2]
        print(f"[INFO] processing image [{name}] {i + 1}/{len(imagePaths)}")

        # load the input image and convert from BGR to RGB for dlib
        image = cv2.imread(imagePath)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x,y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        # we are assuming the the boxes of faces are the SAME FACE or SAME PERSON
        boxes = face_recognition.face_locations(rgb_image, model=detection_method)

        # compute the facial embedding for the face
        # creates a vector of 128 numbers representing the face
        encodings = face_recognition.face_encodings(rgb_image, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    e = time.time()
    print(f"Encoding dataset took: {(e - s) / 60} minutes")
    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}

    if os.path.exists(encodings_file):
        # then unpickle and add to the file
        with open(encodings_file, mode="rb") as opened_file:
            results = pickle.load(opened_file)
            data['encodings'].extend(results['encodings'])
            data['names'].extend(results['names'])

    # write new full set of encodings
    f = open(encodings_file, "wb")
    f.write(pickle.dumps(data))
    f.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset directory.  If there are multiple directories all subdirectories will be encoded")
    ap.add_argument("-e", "--encodings-file", required=True,
                    help="path to serialized pickle file of facial encodings.  If the file exists, new encodings will be added.  Otherwise the file will be created")
    ap.add_argument("-m", "--detection-method", type=str, required=False, default='hog',
                    help="face detection model to use: either 'hog' or 'cnn' ")
    ap.add_argument("-r", "--remove-existing-encodings", type=bool, required=False, default=False,
                    help="Remove existing encodings if they exist")

    args = vars(ap.parse_args())

    if args['remove_existing_encodings']:
        encodings_path = Path(args['encodings_file'])
        if encodings_path.exists():
            encodings_path.unlink()

    create_encodings(args['dataset'], args['encodings_file'], args['detection_method'])
