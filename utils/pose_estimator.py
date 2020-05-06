''' utils to generate pose estimators for head, upper body, lower body and feet.
    Alphapose output format at https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/output.md
    Format: COCO with 17 kps
'''
import json


def json_to_dict(json_path):
    '''Convert json file to dictionary'''
    f = open(json_path)
    data = json.load(f)
    return data

def extract_ids(json_path):
    ids = []
    data = json_to_dict(json_path)
    for dic in data:
        ids.append(dic["image_id"])
    return ids


def extract_kps(img_name, json_path):
    ''' Extract ALL keypoints from json
        Args:
            img_name: image name to visualize
            json_path: path to Alphapose output
        Return:
            kps: keypoints as a list
        '''
    data = json_to_dict(json_path)
    for dic in data:
        if dic["image_id"] == img_name:
            kps = dic["keypoints"]
    return kps

def get_kps_idx(img_name, json_path, indices):
    ''' Extract keypoints with index in indices list '''
    kps = extract_kps(img_name,json_path)
    print(len(kps))
    kp = kps[3*indices:3*indices+3]
    return kp

def get_head():
    ''' Get head estimator and block the rest image with black color '''
    return None

def get_upper():
    return None

def get_lower():
    return None

def get_feet():
    return None

def get_estimator():
    ''' Get six estimators for image_name '''
    return None
def get_all_estimator():
    ''' Get six estimators for all images '''
    return None
