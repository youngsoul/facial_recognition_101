# import the necessary packages
import face_recognition
import pickle
import argparse
import cv2
from imutils.video import VideoStream
import imutils
import time

"""

"""
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings-file", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-m", "--detection-method", type=str, required=False, default='hog',
                help="face detection model to use: either 'hog' or 'cnn' ")
ap.add_argument("-d", "--distance-tolerance", type=float, required=False, default=0.55,
                help="Distance tolerance used determine if there is a facial encoding match")

args = vars(ap.parse_args())

tolerance = args['distance_tolerance']

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args['encodings_file'], "rb").read())

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(0.3)

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

# loop over frames from the vdeo file stream
while True:
    frame = vs.read()

    # convert the input frame from BGR to RGB then resize it to have a width
    # of 750px (to speedup processing)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_image = imutils.resize(rgb_image, width=750)
    r = frame.shape[1] / float(rgb_image.shape[1])

    # detect the (x,y)-coordinates of the bounding boxes corresponding to each face in the
    # input frame, then compute the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb_image, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb_image, boxes)

    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(data['encodings'], encoding, tolerance=tolerance)
        name = "Unknown"

        # check to see if we have found any matches
        if True in matches:
            # find the indexes of all matched faces then initialize a dictionary to count
            # the total number of times each face was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for each recognized face face
            for i in matchedIdxs:
                name = data['names'][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of votes: (notes: in the event of an unlikely
            # tie, Python will select first entry in the dictionary)
            name = max(counts, key=counts.get)
        names.append(name)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # rescale the face coordinates
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
