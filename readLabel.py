import cv2
import pytesseract
import numpy as np
import textdistance
from skimage.metrics import structural_similarity as compare_ssim
import re

pytesseract.pytesseract.tesseract_cmd=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def find_bbox(img, thres_for_minumum_area=0.01, thres_for_inside_points=0.2):
    original = img.copy()
    lower_bound, upper_bound = np.array([0, 0, 170]), np.array([70, 70, 255])
    img = cv2.inRange(img, lower_bound, upper_bound)
    result = []
    area = img.shape[0] * img.shape[1]
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > thres_for_minumum_area * area:
            ROI = img[y:y + h, x:x + w]
            zeros = 0
            for row in ROI:
                zeros += row.tolist().count(0)
            if zeros > thres_for_inside_points * w * h:
                obj = original[y:y + h, x:x + w]
                result.append((obj.copy(), int(y + h / 2), int(x + w / 2), ROI))
    return result


def get_list_of_objects(path):
    list_of_objects = {}
    video = cv2.VideoCapture(path)
    ret, org = video.read()
    frame_num = 0
    
    while ret:
        target = find_bbox(org)
        if len(target) > 0:
            list_of_objects[frame_num] = target
        frame_num += 1
        ret, org = video.read()
    return list_of_objects


def get_ssim_matrix(lst1, lst2):
    result = []
    for image1 in lst1:
        sub_list = []
        for image2 in lst2:
            coefficient = min(image1.shape[1] * image1.shape[0], image2.shape[1] * image2.shape[0]) / max(
                image1.shape[1] * image1.shape[0], image2.shape[1] * image2.shape[0])
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            score, diff = compare_ssim(gray1, gray2, full=True)
            sub_list.append(coefficient * score)
        result.append(sub_list)
    return result


def search_closed_label(input_string, labels=['car', 'truck', 'bicycle', 'train', 'person', 'bus', 'motorcycle']):
    scores = [textdistance.levenshtein(input_string, i) for i in labels]
    return labels[np.argmin(scores)]


def get_label(image):
    a = np.array(image)
    columns_keep = []
    for column_index in range(a.shape[1]):
        column = a[:, column_index]
        mask = column > 200
        num = np.sum(mask)
        columns_keep.append(num)
    columns_keep = columns_keep > np.average(np.array(columns_keep))
    columns_keep = np.where(columns_keep)[0]
    min_index, max_index = np.min(columns_keep), np.max(columns_keep)
    image = a[:, min_index:max_index + 1]
    text = pytesseract.image_to_string(image, config='--psm 6 --dpi 300')
    text = re.sub('[^a-zA-Z]+', '', text)
    label = search_closed_label(text)
    return label


def merge_objects(objects, thres_for_similarity=0.18, thres_for_appearance=0.4):
    list_of_tracking_objects = []

    if len(objects.keys()) == 0:
        return list_of_tracking_objects

    first_key = min(list(objects.keys()))
    sorted_keys = sorted(list(objects.keys()))

    for item in objects[first_key]:
        list_of_tracking_objects.append(
            {"image": item[0], "trajectory": [(item[2], item[1])], "start_frame": first_key, "end_frame": first_key,
             "bbox_image": item[3][:19]})

    if len(objects.keys()) > 1:
        for key in sorted_keys[1:]:
            objects_in_this_frame = objects[key]
            imgaes_in_this_frame = [i[0][22:-8, 8:-8] for i in objects_in_this_frame]
            list_of_different_objects = [i['image'][22:-8, 8:-8] for i in list_of_tracking_objects]

            matrix = np.array(get_ssim_matrix(list_of_different_objects, imgaes_in_this_frame))

            if len(matrix) == 0:
                return list_of_tracking_objects

            current_max = np.max(matrix)
            status = [True for i in range(len(imgaes_in_this_frame))]

            while current_max >= thres_for_similarity:
                obj_id, img_id = np.unravel_index(matrix.argmax(), matrix.shape)

                if -1 not in matrix[obj_id] and -1 not in matrix[:, img_id]:
                    temp_trajectory = list_of_tracking_objects[obj_id]["trajectory"]
                    list_of_tracking_objects[obj_id]["trajectory"].append(
                        (objects_in_this_frame[img_id][2], objects_in_this_frame[img_id][1]))
                    list_of_tracking_objects[obj_id]["end_frame"] = key
                    status[img_id] = False

                matrix[obj_id, img_id] = -1
                current_max = np.max(matrix)

            for i in range(len(status)):
                if status[i]:
                    list_of_tracking_objects.append({"image": objects_in_this_frame[i][0], "trajectory": [
                        (objects_in_this_frame[i][2], objects_in_this_frame[i][1])], "start_frame": key,
                                                     "end_frame": key, "bbox_image": objects_in_this_frame[i][3][:19]})

    result = []

    max_len = max([obj["end_frame"] - obj["start_frame"] for obj in list_of_tracking_objects])

    for obj in list_of_tracking_objects:
        if obj["end_frame"] - obj["start_frame"] > thres_for_appearance * max_len:
            obj['label'] = get_label(obj["bbox_image"])
            del obj['bbox_image']
            result.append(obj)

    return result


#path="B_2.jpg"
#image = cv2.imread(path)
#label=get_label(image)
#print(label)

path="https://igct.s3.amazonaws.com/5fee675583dbb051a5aa4b16_1626185723.mp4" #write video path here
objects=get_list_of_objects(path)
objects=merge_objects(objects)
print("Total number of objects: ",len(objects))
i=1
for obj in objects:
  cv2.imwrite("A_"+str(i)+".jpg",obj['image'])
  print("image name: "+"A_"+str(i)+".jpg ","trajectory: ",obj['trajectory'],"Appearance: ",str(obj["start_frame"])+"-"+str(obj["end_frame"]),"label: "+obj['label'])
  i+=1
