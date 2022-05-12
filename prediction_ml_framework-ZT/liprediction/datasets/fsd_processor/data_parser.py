import numpy as np
from datasets.database.sample_database import SampleDataset


class Parser():

    def __init__(self, config):
        self.config = config['fsd_processor']
        self.lane_max_vector_num = self.config['default_lane_vector_num']
        self.lane_vector_dim = self.config['lane_vector_dim']
        self.obs_max_vector_num = self.config['default_history_num']
        self.obs_vector_dim = self.config['obs_vector_dim']

    def parser(self, input_dir):
        input_database = SampleDataset(input_dir, 'lmdb')

        seq_num_map_dict, curr_seq_num_dict = self.map_parser(input_database)
        seq_num_obs_id_dict, obs_id_track_dict = self.obstacle_parser(input_database)

        return seq_num_obs_id_dict, obs_id_track_dict, seq_num_map_dict, curr_seq_num_dict

    def obstacle_parser(self, input_database):
        seq_num_obs_id_dict = {}
        obs_id_track_dict = {}
        for i in range(len(input_database)):
            sample_name = input_database.get_sample_name(i)
            sample_pb = input_database.get_sample(sample_name)
            seq_num = sample_pb.seq_num
            obstacles_pb = sample_pb.obstacle_features
            obs_id_list, obs_info_list = self.obs_parser(obstacles_pb, seq_num)
            assert seq_num not in seq_num_obs_id_dict
            seq_num_obs_id_dict[seq_num] = obs_id_list
            max_seq_num = max(list(seq_num_obs_id_dict.keys()))
            assert seq_num > max_seq_num
            for obs_id, obs_info in zip(obs_id_list, obs_info_list):
                obs_id_track_dict.setdefault(obs_id, {}).update({seq_num: obs_info})

        return seq_num_obs_id_dict, obs_id_track_dict

    def map_parser(self, input_database: SampleDataset):
        seq_num_map_dict = {}
        curr_seq_num_dict = {}
        for i in range(len(input_database)):
            sample_name = input_database.get_sample_name(i)
            sample_pb = input_database.get_sample(sample_name)
            map_polyline = sample_pb.map_polyline
            curr_seq_num = sample_pb.seq_num
            seq_num = map_polyline.seq_num
            curr_seq_num_dict[curr_seq_num] = seq_num
            if not map_polyline.is_update:
                continue
            assert seq_num not in seq_num_map_dict
            max_seq_num = max(list(seq_num_map_dict.keys()))
            assert seq_num > max_seq_num
            lane_array_dict = self.lane_parser(map_polyline.lanes)
            bound_array_dict = self.bound_parser(map_polyline.lane_boundarys)
            seq_num_map_dict[seq_num] = [lane_array_dict, bound_array_dict]
        return seq_num_map_dict, curr_seq_num_dict

    def lane_parser(self, lane_pb):
        lane_polyline_dict = {}
        for lane_polyline in lane_pb:
            polyline_id = lane_polyline.id
            count = 0
            lane_vectors = []
            for vector in lane_polyline.lane_vector:
                vp = vector.vec.points
                assert len(vp) == 2
                vector_fea = [
                    vp[0].x, vp[0].y, vp[1].x, vp[1].y, vector.heading, count, vp.polyline_type, vector.is_virtual,
                    vector.lane_type, vector.speed_limit
                ]
                vector_fea = np.array(vector_fea)
                lane_vectors.append(vector_fea)
                count += 1
            lane_vectors = np.array(lane_vectors)
            assert lane_vectors.shape[0] <= self.lane_max_vector_num and lane_vectors.shape[1] < self.lane_vector_dim
            lane_vectors_array = np.zeros((self.lane_max_vector_num, self.lane_vector_dim))
            lane_vectors_array[:lane_vectors.shape[0], :lane_vectors.shape[1]] = lane_vectors
            lane_vectors_array[:lane_vectors.shape[0], -1] = 1.0
            lane_polyline_dict[polyline_id] = lane_vectors_array

        return lane_polyline_dict

    def bound_parser(self, bound_pb):
        bound_polyline_dict = {}
        for bound_polyline in bound_pb:
            polyline_id = bound_polyline.id
            count = 0
            bound_vectors = []
            for vector in bound_polyline.lane_boundary_vec:
                vp = vector.vec.points
                assert len(vp) == 2
                vector_fea = [vp[0].x, vp[0].y, vp[1].x, vp[1].y, vector.heading, count, vp.polyline_type, vector.type]
                vector_fea = np.array(vector_fea)
                bound_vectors.append(vector_fea)
                count += 1
            bound_vectors = np.array(bound_vectors)
            assert bound_vectors.shape[0] <= self.lane_max_vector_num and bound_vectors.shape[1] < self.lane_vector_dim
            lane_vectors_array = np.zeros((self.lane_max_vector_num, self.lane_vector_dim))
            lane_vectors_array[:bound_vectors.shape[0], :bound_vectors.shape[1]] = bound_vectors
            lane_vectors_array[:bound_vectors.shape[0], -1] = 1.0
            bound_polyline_dict[polyline_id] = lane_vectors_array

        return bound_polyline_dict

    def obs_parser(self, obstacle_pb, seq_num):
        obs_id_list = []
        obs_state_list = []
        for obs_state in obstacle_pb:
            obs_id = obs_state.id
            obs_id_list.append(obs_id)
            assert seq_num == obs_state.seq_num

            pose = np.array([obs_state.x, obs_state.y, obs_state.heading])
            polyline_info = None
            if obs_state.HasField('polyline_info'):
                polyline_info = obs_state.polyline_info
            history_array = np.zeros((self.obs_max_vector_num, self.obs_vector_dim))

            for obs_vec in obs_state.obs_vec:
                assert obs_vec.seq_num <= seq_num
                assert len(obs_vec.vec.points) == 2
                vp = obs_vec.vec.points
                frame_dist = seq_num - obs_vec.seq_num
                obs_fea = [
                    vp[0].x, vp[0].y, vp[1].x, vp[1].y, obs_vec.heading, frame_dist, obs_vec.type, obs_vec.velocity.x,
                    obs_vec.velocity.y, obs_vec.velocity_converged, obs_vec.speed, obs_vec.acc.x, obs_vec.acc.y,
                    obs_vec.scalar_acc, obs_vec.length, obs_vec.width, obs_vec.height, obs_vec.lf.x, obs_vec.lf.y,
                    obs_vec.rf.x, obs_vec.rf.y, obs_vec.lr.x, obs_vec.lr.y, obs_vec.rr.x, obs_vec.rr.y,
                    obs_vec.dist_to_adc, obs_vec.adc_theta, obs_vec.fusion_type, obs_vec.is_interpolation_vec
                ]
                obs_fea = np.array(obs_fea)
                obs_fea[-1] = 1.0
                history_array[frame_dist][:obs_fea.shape[0]] = obs_fea
            obs_state_list.append([history_array, polyline_info, pose])

        return obs_id, obs_state_list
