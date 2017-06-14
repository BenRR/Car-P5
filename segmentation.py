import train
import cv2
import numpy as np
from scipy.ndimage.measurements import label

VERT_FROM = 380
VERT_TO = 660
BB_THRESHOLD=0.99
HEAT_THRESHOLD=15


def region_of_interest(frame):
    return frame[VERT_FROM:VERT_TO, 500:1280, :]


def load_seg_model(model_file):
    return train.create_model(True, (None, None, 3), model_file)


def seg(frame, model_file):
    model = load_seg_model(model_file)
    return seg_by_model(frame, model)


def seg_by_model(frame, model):
    image = region_of_interest(frame)
    return model.predict(image.reshape(1, image.shape[0], image.shape[1], image.shape[2]))


def find_bboxes(segs, threshold=BB_THRESHOLD, conv_size=8, unit_size=64):
    hs, vs = np.meshgrid(np.arange(segs.shape[2]), np.arange(segs.shape[1]))
    x_scale = (hs[segs[0, :, :, 0] > threshold])
    y_scale = (vs[segs[0, :, :, 0] > threshold])
    boxes = []

    for i, j in zip(x_scale, y_scale):
        x = i * conv_size + 500
        y = j * conv_size + VERT_FROM
        boxes.append([x, y, x + unit_size, y + unit_size])
    return np.array(boxes)


def group_bboxes(bboxes, threshold=9, diff=.1):
    return cv2.groupRectangles(rectList=bboxes.tolist(), groupThreshold=threshold, eps=diff)


def draw_boxes(img, bboxes):
    for box in bboxes:
        cv2.rectangle(img, (box[0], box[1]),
                      (box[2], box[3]), (0, 0, 255), 4)
    return img


def heatmap(img, bboxes):
    heat = np.zeros((img.shape[0], img.shape[1]))
    for box in bboxes:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        # print(box)
        heat[box[1]:box[3], box[0]:box[2]] += 1
    return heat


def heatlabel(heat, threshold=HEAT_THRESHOLD):
    heat[heat <= threshold] = 0
    return label(heat)


def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


def grid(img, segs):
    bboxes = find_bboxes(segs)

    return draw_boxes(img, bboxes)


def grid_all(img, segs):
    bboxes = find_bboxes(segs)
    heat = heatmap(img, bboxes)
    labels = heatlabel(heat)

    return draw_labeled_bboxes(img, labels)


if __name__ == '__main__':
    feature_map = seg("test_images/test1.jpg", "weights-improvement-09-0.49.h5")
    print(feature_map.shape)

