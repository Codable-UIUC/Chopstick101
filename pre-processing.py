import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def store_data(path: str, data: np.ndarray) -> None:
    """
    Stores given data in pkl.

    Parameters
    ----------
    filepath : str
    The path of file to store.

    filename : str
    The title of file to store.

    data : ndarray[ndarray[int]], (# frames, 21*3)
    2D ndarray data to store. Rows are each frames, and coulmns are x, y, z of each landmark
    """
    with open(path,'wb') as f:
        pickle.dump(data, f)

def processing(filename: str) -> np.ndarray:
    """
    Performs data pre-processing of given file.
    Data is taken from video with 24~30 fps video.
    Starting at 30th frame, taken for 90 frames onward.

    Parameters
    ----------
    filename: str
        The path to file

    Returns
    -------
    data : np.ndarray, (90, 21*3)
        2D ndarray data from each frame. There are 21 landmarks and 3 coordinates(x,y,z) per frame.
        90 frames are taken independent of the video fps(24~30).
    """
    ### For webcam input: ###
    cap = cv2.VideoCapture(filename)
    ### output source to store overlayed video ###
    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
    # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        cnt = 0
        data = []
        while cap.isOpened():
            frame = []
            success, image = cap.read()
            if not success:
                print("Completed.")
                # continue                    ### for webcam/live
                break                       ### for video
            
            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            ### Draw the hand annotations on the image. ###
            # image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks and 30 < cnt:
                for i in (results.multi_hand_landmarks[0].landmark):
                    ## num of frames x 21 landmarks x 3 properties (N, 21, 3)
                    frame.extend([i.x, i.y, i.z])
                data.append(frame)

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            ### FOR RECORDING FRAMES; used with 'out' variable above ###
            # cv2.imwrite(f'videoframe/new{cnt}.png', image)
            # out.write(image)
            if 120 == cnt:
                break
            cnt += 1

    cap.release()
    return np.array(data)

def main() -> None:
    """
    Main function to run the pre-processing task.
    """
    path_video_true = r".\input_data\Video\True"
    path_video_false = r".\input_data\Video\False"
    path_frame_true = r".\input_data\Frames\True"
    path_frame_false = r".\input_data\Frames\False"
    paths = ((path_video_true, path_frame_true), (path_video_false, path_frame_false))

    for path_video, path_frame in paths:
        iter = os.scandir(path=path_video)          # iterates through all files in the path
        for file in iter:
            filename = file.name
            print(f"{filename} is started.")
            start = time.time()
            video_filepath = path_video + '\\' + filename       # get path to each video
            data = processing(video_filepath)                   # generate data of each video
            frame_filepath = path_frame + '\\' + filename.split(".")[0] + ".pkl"    # make path to new pkl file
            store_data(frame_filepath, data)                    # store data at given path
            print(f"{filename} is completed in {round(time.time()-start, 2)} sec.")

if __name__ == '__main__':
    main()