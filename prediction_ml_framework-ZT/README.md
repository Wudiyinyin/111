# prediction_ml_framework

Prediction Machine Learning Framework


## Project Formatting

> reference to : https://li.feishu.cn/docs/doccnMjJZelpeZBLjPEEmQyPt3f

# Install

```
pip install -r requirements/requirements.txt
```

> Note: `requirements.txt` is exported by following comand:
```
Please confirm python version is 3.8

pip install pipreqs
pipreqs ./ --force
```

> P.S. Using TUNA source will accelerate downloading repos, reference: https://mirrors.tuna.tsinghua.edu.cn/help/pypi/

# Run

```
source path.bash

# plot new sample
python liprediction/datasets/database/utils/plot_new_sample.py -db ~/work/Waymoo/train

# plot old sample
python liprediction/datasets/database/utils/plot_sample.py ~/work/Argoverse/train ~/work/Argoverse/train.txt

# plot transform_sample
python liprediction/datasets/transform/sample_transform/transform_upsample.py models/densetnt/config/upsample.yaml ~/Desktop/

# model infer test
python liprediction/models/densetnt/densetnt.py models/densetnt/config/upsample.yaml

# train
## multi gpus in single machine
python liprediction/apis/train.py --config liprediction/models/densetnt/config/dense_tnt.yaml --saved_model ./result/dense_tnt --log_dir ./log/dense_tnt --gpu -1

## multi gpus in multi nodes, run script in each node
MASTER_ADDR=172.21.32.5 MASTER_PORT=8889 NODE_RANK=0 python liprediction/apis/train.py --config liprediction/models/densetnt/config/dense_tnt.yaml --saved_model ./result/dense_tnt --log_dir ./log/dense_tnt --gpu 8 --node 2
MASTER_ADDR=172.21.32.5 MASTER_PORT=8889 NODE_RANK=1 python liprediction/apis/train.py --config liprediction/models/densetnt/config/dense_tnt.yaml --saved_model ./result/dense_tnt --log_dir ./log/dense_tnt --gpu 8 --node 2

# eval
python liprediction/apis/eval.py --config models/densetnt/config/upsample.yaml -g 0 --checkpoint log/upsample/version_1/checkpoints/epoch\=6-step\=44764.ckpt

# plot bad case
# set models/densetnt/config/dense_tnt.yaml ['show _badcase']['plot_badcase'] to True
python liprediction/apis/eval.py --config liprediction/models/densetnt/config/dense_tnt.yaml -g 1 --checkpoint liprediction/checkpoint-epoch=0004-val_loss=4.21655.ckpt

# export
python liprediction/apis/export.py --config liprediction/models/densetnt/config/dense_tnt.yaml --checkpoint ./checkpoint-epoch.ckpt --type onnx --path ./output

## Visualize Model Structure
netron MODEL_NAME.onnx

# tensorboard dev upload
cd ${PATH_TO_LOG_DIR}
tensorboard dev upload --logdir ./ --name "My latest experiment" --description "Simple comparison of several hyperparameters"

# benchmark
python liprediction/apis/benchmark.py --input /home/kunzhan/Downloads/test_submit.pb --gt_file /media/kunzhan/LocalFileSystem/Waymo_Dataset/scene_info_dict.pkl

python scripts/tfrecord_parser/submission_waymo.py --input_pkl_dir ~/Desktop/test_output/ --output_pb_file ~/Desktop/test.pb --gt_file /mnt/data/Waymo_Dataset/scenario/validation/scene_info_dict.pkl --task_type single --email your@gmail.com
```
