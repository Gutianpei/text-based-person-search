import json


''' utils to generate pose estimators for head, upper body, lower body and feet.
    Alphapose output format at https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/output.md
'''

def extract_kp(img_name, json_path):
    '''Extract ALL keypoints from json'''

    return kps

def vis_kps(kp_idx, img_name, json_path):
    '''Visualize keypoints

    Args:
        kps_idx (int list): keypoints indice in Alphapose output json file
        img_name: image name to visualize
        json_path: path to Alphapose output
    Return:
        img_vis: image with keypoints visualization
    '''
    kps = extract_kp(img_name, json_path)
    kp = 
    return img_vis
