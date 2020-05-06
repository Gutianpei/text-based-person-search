import pose_estimator


json_path = "/home/gutianpei/prg/text-based-person-search/demo_output/alphapose-results.json"

ids = pose_estimator.extract_ids(json_path)
id = ids[0]

print(pose_estimator.get_kps_idx(id,json_path,9))
