import cv2
import mediapipe as mp

def get_hands(image, padding=0):
    mp_hands = mp.solutions.hands
    mp_model = mp_hands.Hands(
    static_image_mode=True, # only static images
    max_num_hands=2, # max 2 hands detection
    min_detection_confidence=0.2) # detection confidence
    
    res = []
    prediction = mp_model.process(image)
    if prediction.multi_hand_landmarks:
        res = [get_bbox_coordinates(x, image.shape, padding) for x in prediction.multi_hand_landmarks]
    return res

def get_bbox_coordinates(detection, image_shape, padding=0):
    """ 
    Get bounding box coordinates for a hand landmark.
    Args:
        handLadmark: A HandLandmark object.
        image_shape: A tuple of the form (height, width).
    Returns:
        A tuple of the form (xmin, ymin, xmax, ymax).
    """
    mp_hands = mp.solutions.hands
    all_x, all_y = [], [] # store all x and y points in list
    for hnd in mp_hands.HandLandmark:
        all_x.append(int(detection.landmark[hnd].x * image_shape[1])) # multiply x by image width
        all_y.append(int(detection.landmark[hnd].y * image_shape[0])) # multiply y by image height

    return min(all_x)-padding, min(all_y)-padding, max(all_x)+padding, max(all_y)+padding # return as (xmin, ymin, xmax, ymax)