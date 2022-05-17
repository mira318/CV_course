import json
import os
import time

import cv2
import numpy as np
import pandas as pd


def detect_pattern(input_path, output_path, return_contour=False, display=False):
    input_image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if input_image is None:
        print('Cant open, path = ', input_path)
        return

    image_to_draw = input_image.copy()
    input_image = cv2.GaussianBlur(input_image, (5, 5), 0)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_image = cv2.adaptiveThreshold(input_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
    input_image = cv2.medianBlur(input_image, 9)
    contours, hierarchy, candidates = get_contours(input_image)
    res_candidates = filter(contours, hierarchy, candidates)

    if display:
        for candidate in res_candidates:
            cv2.drawContours(image_to_draw, contours, candidate, (0, 255, 0), 3, cv2.LINE_8, hierarchy, 0)
        cv2.imwrite(output_path, image_to_draw)

    if return_contour:
        return (contours[i] for i in res_candidates)


def get_contours(input_image):
    detected_edges = cv2.Canny(input_image, 80, 120, 7)
    mask = (detected_edges != 0).astype(input_image.dtype) * 255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    all_candidates, right_candidates = [], []

    for i in range(hierarchy.shape[0]):
        all_candidates.append(i)
        child = i
        layers = 0
        while hierarchy[child, 2] != -1:
            child = hierarchy[child, 2]
            layers += 1

        if hierarchy[child, 2] != -1:
            layers += 1
        if 4 <= layers <= 6:
            right_candidates.append(i)

    return contours, hierarchy, right_candidates

def filter(contours, hierarchy, candidates):
    right_candidates, right_layers = [], []
    ideal_ratio01 = 49 / 25
    ideal_ratio12 = 25 / 9

    for candidate in candidates:
        child = hierarchy[candidate, 2]
        layers = [candidate]
        while child != -1:
            layers.append(child)
            child = hierarchy[child, 2]

        approx = cv2.approxPolyDP(contours[candidate], 0.05 * cv2.arcLength(contours[candidate], True), True)

        if len(approx) == 4:
            w = np.sqrt(np.sum((approx[0, 0] - approx[1, 0]) ** 2).astype(float))
            h = np.sqrt(np.sum((approx[1, 0] - approx[2, 0]) ** 2).astype(float))
            cont0 = contours[layers[0]]
            cont1 = contours[layers[1]]
            cont2 = contours[layers[2]]
            ratio01 = cv2.contourArea(cont0) / cv2.contourArea(cont1)
            ratio12 = cv2.contourArea(cont1) / cv2.contourArea(cont2)

            if abs(w - h) / min(w, h) < 0.25:
                if (0.5 * ideal_ratio01 < ratio01 < 2 * ideal_ratio01) or \
                        (0.5 * ideal_ratio12 < ratio12 < 2 * ideal_ratio12):
                    right_candidates.append(candidate)
                    right_layers.append(layers)

    identical = set()
    for i, layers1 in zip(right_candidates, right_layers):
        for j, layers2 in zip(right_candidates, right_layers):
            if (i != j) and (layers1[0] in layers2[1:]):
                identical.add(i)

    res_candidates = [candidate for candidate in right_candidates if candidate not in identical]
    return res_candidates


def count_iou(x_min1, x_max1, y_min1, y_max1, x_min2, x_max2, y_min2, y_max2):
    overlap_s = max(0, min(x_max1, x_max2) - max(x_min1, x_min2)) * \
                max(0, min(y_max1, y_max2) - max(y_min1, y_min2))

    union_s = ((x_max1 - x_min1) * (y_max1 - y_min1)) + \
              ((x_max2 - x_min2) * (y_max2 - y_min2)) - overlap_s

    return overlap_s / union_s


def compare_with_labels(detected, real):
    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(len(real)):
        max_iou = 0
        for j in range(len(detected)):
            iou = count_iou(
                detected[j][0], detected[j][1], detected[j][2], detected[j][3],
                real[i][0], real[i][1], real[i][2], real[i][3]
            )
            max_iou = max(iou, max_iou)

        if max_iou > 0.5:
            tp += 1
        else:
            tn += 1

    fp = len(detected) - tp
    fn = len(real) - tp
    return tp, tn, fp, fn


def prepare_data(path_csv, path_set1, path_set2):
    set1_df = pd.read_csv(path_csv + '/task6_set1_labels.csv')
    set2_df = pd.read_csv(path_csv + '/task6_set2_labels.csv')
    cleared_df = pd.DataFrame()
    file, xmin, xmax, ymin, ymax, width, height = [], [], [], [], [], [], []

    for row in range(len(set1_df)):
        f = path_set1 + '/' + set1_df['image'][row].split('-')[1]
        L = json.loads(set1_df['label'][row])
        for label in L:
            xmin.append(label['x'] * label['original_width'] / 100.0)
            xmax.append((label['x'] + label['width']) * label['original_width'] / 100.0)
            ymin.append(label['y'] * label['original_height'] / 100.0)
            ymax.append((label['y'] + label['height']) * label['original_height'] / 100.0)
            width.append(label['original_width'])
            height.append(label['original_height'])
            file.append(f)

    for row in range(len(set2_df)):
        f = path_set2 + '/' + set2_df['image'][row].split('-')[1]
        L = json.loads(set2_df['label'][row])
        for label in L:
            xmin.append(label['x'] * label['original_width'] / 100.0)
            xmax.append((label['x'] + label['width']) * label['original_width'] / 100.0)
            ymin.append(label['y'] * label['original_height'] / 100.0)
            ymax.append((label['y'] + label['height']) * label['original_height'] / 100.0)
            width.append(label['original_width'])
            height.append(label['original_height'])
            file.append(f)

    cleared_df['file'] = file
    cleared_df['xmin'] = xmin
    cleared_df['xmax'] = xmax
    cleared_df['ymin'] = ymin
    cleared_df['ymax'] = ymax
    cleared_df['width'] = width
    cleared_df['height'] = height
    return cleared_df


def get_real(file_name, data_df):
    part_data = data_df.where(data_df['file'] == file_name).dropna(how='any', axis=0)
    real_labels = part_data[['xmin', 'xmax', 'ymin', 'ymax']].values
    return real_labels


def run_on_labeled(csv_root, sets_root, output_path):
    images_df = prepare_data(csv_root, sets_root + '/TestSet1', sets_root + '/TestSet2')
    tp, tn, fp, fn = 0, 0, 0, 0
    times = []
    for f in np.unique(images_df['file']):
        start = time.time()
        contours = detect_pattern(f, output_path + '/' + f, return_contour=True, display=True)
        times.append(time.time() - start)
        detected, real = [], []
        for contour in contours:
            xmin, ymin = 1e9, 1e9
            xmax, ymax = -1, -1
            for j, point in enumerate(contour):
                xmin = min(xmin, point[0][0])
                xmax = max(xmax, point[0][0])
                ymin = min(ymin, point[0][1])
                ymax = max(ymax, point[0][1])
            detected.append([xmin, xmax, ymin, ymax])

        real = get_real(f, images_df)
        tp_curr, tn_curr, fp_curr, fn_curr = compare_with_labels(detected, real)
        tp += tp_curr
        tn += tn_curr
        fp += fp_curr
        fn += fn_curr
    print('precision = ', tp / (tp + fp))
    print('recall = ', tp / (tp + fn))
    print('avg_time = ', np.mean(np.array(times)), ' sec per image')


def run_on_test(set_root, output_root):
    for f in os.listdir(set_root):
        detect_pattern(set_root + '/' + f, output_root + '/' + f, display=True)


if __name__ == '__main__':
    run_on_labeled('./', './images', './output_train')
    run_on_test('./images/TestSet3', './output_test')
