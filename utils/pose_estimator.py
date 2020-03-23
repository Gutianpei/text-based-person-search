''' utils to generate pose estimators for head, upper body, lower body and feet.
    Alphapose output format at https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/output.md
'''

def json_to_dict(json_path):
    '''Convert json file to dictionary '''

    return data

def extract_kps(img_name, json_path):
    ''' Extract ALL keypoints from json
        Args:
            img_name: image name to visualize
            json_path: path to Alphapose output
        Return:
            kps: keypoints as a list
        '''
    return kps

def get_kps_idx(img_name, json_path, indices):
    ''' Extract keypoints with index in indices list '''

    return kps

def get_head():
    ''' Get head estimator and block the rest image with black color '''


def get_upper():


def get_lower():


def get_feet():


def get_estimator():
    ''' Get four estimators for image_name '''

def get_all_estimator():
    ''' Get four estimators for all images '''
