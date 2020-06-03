# text-based-person-search

1. Install Alphapose:
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md



python3 scripts/demo_inference.py --indir ../text-based-person-search/data/ --outdir ../text-based-person-search/demo_output/ --vis --save_img --cfg configs/test.yaml --checkpoint pretrained_models/fast_421_res152_256x192.pth



# Google Cloud Tutorial

启动服务器（开始计费）

gcloud compute instances start instance-1


关闭服务器（结束计费）

gcloud compute instances stop instance-1

连接服务器

gcloud compute ssh --project text-based-re-id --zone asia-east1-a instance-1
Password: Gu122300

连接jupyter



改权限

sudo chown -R gutianpei:gutianpei anaconda3

sudo chmod -R 777 ./brucegu


Note：

务必保持directory整洁避免误操作，比如训好的model单独放一个文件夹

务必给mode有意义的名字，带实验编号，实时更新training的google sheet
