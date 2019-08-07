from imutils.video import VideoStream
import time
import cv2
from video_hog import create_HOG_image
import face_recognition
import argparse

"""
Script to display the hog image from the video cam capture with face detected box

"""

def detect_mark_faces(image, hog_image):
    boxes = face_recognition.face_locations(image, model='hog')
    for (top, right, bottom, left) in boxes:
        # draw the predicted face name on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.rectangle(hog_image, (left, top), (right, bottom), (255, 255, 255), 2)


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

        hog_image = create_HOG_image(frame)
        detect_mark_faces(frame, hog_image)
        cv2.imshow("Original", frame)
        cv2.imshow("HOG", hog_image)

        key = cv2.waitKey(3) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        if args['display_time'] > 0 and (time.time() - start) > args['display_time']:
            break
