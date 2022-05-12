# Convert TFRecord to Serialized Protobuf

This scripts can convert Waymo Prediction Dataset Scenario data into Serialized Protobuf without tfrecord.

## Requirement

- Waymo-Open-Dataset-TF >= 2.6
> pip3 install waymo-open-dataset-tf-2-6-0 --user

- joblib
> pip install joblib

## Streaming Muliple Messages

Data Struct = [4-bytes, pb.SerializeToString(), 4-bytes, pb.SerializeToString(), ...]

- 4 bytes : This is the length of PB serialized data, which is Big Order saved in memory and no-signed bytes
- pb serialized data : This is the main protobuf in each frame, which is the scenario_pb2 in waymo prediction dataset
