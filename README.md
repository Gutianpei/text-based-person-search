# text-based-person-search

1. Install Alphapose:
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md



python3 scripts/demo_inference.py --indir ../text-based-person-search/data/ --outdir ../text-based-person-search/demo_output/ --vis --save_img --cfg configs/test.yaml --checkpoint pretrained_models/fast_421_res152_256x192.pth



# TODO:

## 1. Loading Data:

- [x] Load raw image with shape (n,img_height, img_width, 3)

- [X] Load raw captions with shape (2n, step_size=99, vector_size) (2 captions for each image)

- [ ] Run Alphapose get all keypoints and crop body parts


## 2. Preprocessing:

- [ ] Standardization
- [X] Word2Vec (Averaging)


## 3. Network:

- [x] Build resnet50 and get output feature vector

- [X] Bi-LSTM
- [ ] Structure (2 directions)
