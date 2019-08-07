from imutils.video import VideoStream
import time
import cv2
import face_recognition
import argparse

"""
Script to display detected face along with some of the encoding values.
"""


def detect_mark_faces(image):
    boxes = face_recognition.face_locations(image, model='hog')
    encodings = face_recognition.face_encodings(image, boxes)
    if len(encodings) > 0:
        encoding = encodings[0]
        encoding_string = f"{encoding[0]:.2f}, {encoding[32]:.2f}, {encoding[64]:.2f}, {encoding[96]:.2f}, {encoding[127]:.2f}"
    else:
        encoding_string = ""
    # print(encoding_string)
    for (top, right, bottom, left) in boxes:
        # draw the predicted face name on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, encoding_string, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--display-time", type=int, required=False, default=-1,
                    help="Amount of time to display the HOG image.  Sometimes 'q' does not work right away")
    args = vars(ap.parse_args())
    start = time.time()

    vs = VideoStream(src=0).start()
    time.sleep(0.2)

    # loop over frames from the vdeo file stream
    while True:
        # grab the frame from the threaded video stream
        frame = vs.read()

        detect_mark_faces(frame)
        cv2.imshow("Original", frame)

        key = cv2.waitKey(3) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        if args['display_time'] > 0 and (time.time() - start) > args['display_time']:
            break
