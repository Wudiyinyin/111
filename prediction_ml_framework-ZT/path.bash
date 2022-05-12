# Copyright (c) 2021 Li Auto Company. All rights reserved.

WORK_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cd liprediction/protos/ && bash generate_pb.sh && cd -

unset PYTHONPATH

# Dont change this order of python path
export PYTHONPATH=${PYTHONPATH}:${WORK_PATH}/liprediction
export PYTHONPATH=${PYTHONPATH}:${WORK_PATH}/liprediction/protos/proto_py
export PYTHONPATH=${PYTHONPATH}:${WORK_PATH}/../artemis_prediction/output/python

# Useless for now
# export PYTHONPATH=${PYTHONPATH}:${WORK_PATH}/liprediction/datasets/database/proto/prediction

echo ${PYTHONPATH}
