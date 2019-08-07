from imutils.video import VideoStream
import time
import cv2
import argparse
from pathlib import Path

def show_frame(frame, count, msg):
    width, height = frame.shape[:2]
    cv2.putText(frame, f"CountDown: {str(count)}: {msg}", (width // 2, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("PhotoBooth", frame)

    key = cv2.waitKey(3) & 0xFF

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to root dataset directory. A directory with name parameter will be created")
    ap.add_argument("-n", "--name", required=True, help="Name of person")
    ap.add_argument("-c", "--count", required=True, help="Number of pictures take")
    args = vars(ap.parse_args())

    vs = VideoStream(src=0).start()
    time.sleep(0.2)

    data_path = Path(args['dataset'])
    data_path =  data_path / args['name']
    data_path.mkdir(parents=True, exist_ok=True)

    # loop over frames from the video file stream
    for i in range(0, int(args['count'])):
        count_down = 3
        print(f"Grab image: {i}")
        count_down_time = time.time()
        frame = vs.read()
        show_frame(frame, str(count_down), f"Picture {i+1} of {args['count']}")
        while True:
            # grab the frame from the threaded video stream
            frame = vs.read()
            if time.time() - count_down_time > 1:
                count_down -= 1
                if count_down > 0:
                    count_down_time = time.time()
                    show_frame(frame, count_down, f"Picture {i+1} of {args['count']}")
                else:
                    new_image_file = data_path / f"{args['name']}_{i}.png"
                    cv2.imwrite(str(new_image_file.absolute()), frame)
                    break
