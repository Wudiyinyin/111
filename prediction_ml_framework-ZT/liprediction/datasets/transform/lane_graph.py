# Copyright (c) 2021 Li Auto Company. All rights reserved.
import copy
import math

import numpy as np
from core.utils.utils import np_vector_norm
from scipy.spatial import KDTree


class LaneGraph(object):

    def __init__(self):
        # node
        self.node_feature = None
        self.node_pre = []
        self.node_suc = []
        self.node_left = []
        self.node_right = []
        self.node_comp = []
        self.node_gid = []

        # component
        self.comp_pre = []
        self.comp_suc = []
        self.comp_son = []

        # kdtree
        self.kdtree = None

    @staticmethod
    def FromProto(proto):
        lane_graph = LaneGraph()

        # process node information
        node_start, node_end = [], []
        for node in proto.lane_nodes:
            attri = node.attribute
            node_start.append([attri.start.x, attri.start.y])
            node_end.append([attri.end.x, attri.end.y])
            lane_graph.node_pre.append(list(node.predecessors))
            lane_graph.node_suc.append(list(node.successors))
            lane_graph.node_left.append(node.left_neighbour)
            lane_graph.node_right.append(node.right_neighbour)
            lane_graph.node_comp.append(node.father_component)

        node_start = np.array(node_start, dtype=np.float32)
        node_end = np.array(node_end, dtype=np.float32)
        node_pos = (node_start + node_end) / 2
        node_delta = (node_end - node_start)
        lane_graph.node_feature = np.hstack((node_pos, node_delta))
        lane_graph.node_gid = range(lane_graph.node_feature.shape[0])

        # process component information
        for comp in proto.lane_components:
            lane_graph.comp_pre.append(list(comp.predecessors))
            lane_graph.comp_suc.append(list(comp.successors))
            lane_graph.comp_son.append(list(comp.son_nodes))

        # construct kdtree
        lane_graph.kdtree = KDTree(node_pos)

        return lane_graph

    def getNearestNodes(self, pos, radius):
        ''' pos can be a point or point array '''
        return self.kdtree.query_ball_point(pos, radius)

    def getNearestKNodes(self, pos, k=1, radius=np.inf, worker_num=1):
        return self.kdtree.query(pos, k, distance_upper_bound=radius, workers=worker_num)

    def getSubGraph(self, ids, construct_kdtree=False, construct_component=False):
        ''' construct sub graph, nodes specified by ids '''

        id_list = list(ids)

        lane_graph = LaneGraph()

        # construct feature
        lane_graph.node_feature = self.node_feature[id_list]

        # construct global_id to local_id map
        id_map = {}
        for local_id, global_id in enumerate(id_list):
            id_map[global_id] = local_id

        # construct connections
        for global_id in id_list:
            lane_graph.node_pre.append([id_map[node_id] for node_id in self.node_pre[global_id] if node_id in id_map])

            lane_graph.node_suc.append([id_map[node_id] for node_id in self.node_suc[global_id] if node_id in id_map])

            left_global_id = self.node_left[global_id]
            lane_graph.node_left.append(-1 if left_global_id not in id_map else id_map[left_global_id])

            right_global_id = self.node_right[global_id]
            lane_graph.node_right.append(-1 if right_global_id not in id_map else id_map[right_global_id])

            lane_graph.node_comp.append(-1)

            lane_graph.node_gid.append(global_id)

        if construct_kdtree:
            lane_graph.kdtree = KDTree(lane_graph.node_feature[:, [0, 1]])

        if construct_component:
            # first find the source node
            sources = []
            for i in range(len(lane_graph.node_pre)):
                if len(lane_graph.node_pre[i]) == 0:
                    sources.append(i)

            # create components from sources
            component = 0
            while sources:
                node = sources.pop()

                # add new component
                lane_graph.comp_pre.append([])
                lane_graph.comp_suc.append([])
                lane_graph.comp_son.append([])

                # set component's pre
                for pre in lane_graph.node_pre[node]:
                    if lane_graph.node_comp[pre] != -1:
                        pre_component = lane_graph.node_comp[pre]
                        lane_graph.comp_suc[pre_component].append(component)
                        lane_graph.comp_pre[component].append(pre_component)

                # process component itself
                while True:
                    lane_graph.comp_son[component].append(node)
                    lane_graph.node_comp[node] = component
                    if not (len(lane_graph.node_suc[node]) != 1 or
                            len(lane_graph.node_pre[lane_graph.node_suc[node][0]]) != 1):
                        node = lane_graph.node_suc[node][0]
                    else:
                        for suc in lane_graph.node_suc[node]:
                            if lane_graph.node_comp[suc] == -1:
                                sources.append(suc)
                        break

                # set component's suc
                for suc in lane_graph.node_suc[node]:
                    if lane_graph.node_comp[suc] != -1:
                        suc_component = lane_graph.node_comp[suc]
                        lane_graph.comp_pre[suc_component].append(component)
                        lane_graph.comp_suc[component].append(suc_component)

                component += 1

        return lane_graph, id_map

    def getAttentionCandidate(self, pos, radius):
        nbs = self.kdtree.query_ball_point(pos, radius)
        dists = np_vector_norm(self.node_feature[nbs][:, [0, 1]] - pos, axis=1)

        # filter by component id
        component_node_map = {}
        for node_id, node_dist in zip(nbs, dists):
            component_id = self.node_comp[node_id]
            if (component_id not in component_node_map) or (component_node_map[component_id][1] > node_dist):
                component_node_map[component_id] = (node_id, node_dist)

        candidates = self._filterCandidates(component_node_map)

        return [candi[1][0] for candi in candidates]

    def constructKDTree(self):
        self.kdtree = KDTree(self.node_feature[:, [0, 1]])

    def _filterCandidates(self, candi_map):
        candi_num = len(candi_map)

        if candi_num < 2:
            return candi_map.items()

        candi_list_merged = [candi for candi in candi_map.items() if self._checkCandidate(candi, candi_map)]

        return candi_list_merged

    def _checkCandidate(self, candi, candi_map):
        curr_component = candi[0]
        curr_distance = candi[1][1]

        # check if candi has alive predecessor
        pre_alive = False
        todo = [curr_component]
        while todo:
            component = todo.pop()

            pre_list = [pre for pre in self.comp_pre[component] if pre in candi_map]

            if len(pre_list) == 0:
                pre_alive = True
                break
            else:
                todo.extend([pre for pre in pre_list if curr_distance < candi_map[pre][1]])

        suc_alive = False
        todo = [curr_component]
        while todo:
            component = todo.pop()

            suc_list = [suc for suc in self.comp_suc[component] if suc in candi_map]

            if len(suc_list) == 0:
                suc_alive = True
                break
            else:
                todo.extend([suc for suc in suc_list if curr_distance <= candi_map[suc][1]])

        return pre_alive and suc_alive

    def isNodeConnected(self, na, nb):
        na_vec = self.node_feature[na, [2, 3]]
        na_heading = math.atan2(na_vec[1], na_vec[0])
        nb_vec = self.node_feature[nb, [2, 3]]
        nb_heading = math.atan2(nb_vec[1], nb_vec[0])

        if getHeadingDiff(na_heading, nb_heading) > math.pi / 2.0:
            return False

        ca, cb = self.node_comp[na], self.node_comp[nb]

        return ca == cb or self.isComponentConnected(ca, cb)

    def isComponentConnected(self, ca, cb):
        '''
        assume ca and cb is not the same component
        '''
        checking_depth = 2

        if ca == cb:
            return True

        # check in predecessors
        todo = [(pre, 1) for pre in self.comp_pre[ca]]
        while todo:
            cc, depth = todo.pop()
            if cc == cb:
                return True
            if depth < checking_depth:
                todo.extend([(pre, depth + 1) for pre in self.comp_pre[cc]])

        # check in successors
        todo = [(suc, 1) for suc in self.comp_suc[ca]]
        while todo:
            cc, depth = todo.pop()
            if cc == cb:
                return True
            if depth < checking_depth:
                todo.extend([(suc, depth + 1) for suc in self.comp_suc[cc]])

        return False

    def getPre(self, nid, max_depth=1):
        pre_list = []
        todo = [(pre, 1) for pre in self.node_pre[nid]]
        while todo:
            cur, depth = todo.pop()
            pre_list.append(cur)
            if depth < max_depth:
                todo.extend([(pre, depth + 1) for pre in self.node_pre[cur]])
        # print(f'pre size:{len(pre_list)}')
        return pre_list

    def getSuc(self, nid, max_depth=1):
        suc_list = []
        todo = [(suc, 1) for suc in self.node_suc[nid]]
        while todo:
            cur, depth = todo.pop()
            suc_list.append(cur)
            if depth < max_depth:
                todo.extend([(suc, depth + 1) for suc in self.node_suc[cur]])
        # print(f'suc size:{len(suc_list)}')
        return suc_list

    def getIntentionCandidate(self, attention_node_index, prediction_back_horizon, prediction_horizon, kinship_horizon):
        # direct key node id
        direct_node_ids = []
        direct_node_ids.append(attention_node_index)
        if self.node_left[attention_node_index] > 0:
            direct_node_ids.append(self.node_left[attention_node_index])
        if self.node_right[attention_node_index] > 0:
            direct_node_ids.append(self.node_right[attention_node_index])

        # brosis and couble key node id
        brosis_node_ids = self.findBrosisKeyNodes(attention_node_index, kinship_horizon)
        couple_node_ids = self.findCoupleKetNodes(attention_node_index, prediction_horizon)

        # history and future key node id
        history_neighbour_node_ids = self.findHistoryNeighborKeyNodes(attention_node_index, prediction_back_horizon)
        future_neighbour_node_ids = self.findFutureNeighborKeyNodes(attention_node_index, prediction_horizon)

        # filter by Connected and length
        brosis_couple_node_ids_exed = []
        for node_id in brosis_node_ids + couple_node_ids:
            if self.isNodeConnected(node_id, attention_node_index):
                continue
            if self.node_left[attention_node_index] > 0 and self.isNodeConnected(node_id,
                                                                                 self.node_left[attention_node_index]):
                continue
            if self.node_right[attention_node_index] > 0 and self.isNodeConnected(
                    node_id, self.node_right[attention_node_index]):
                continue
            brosis_pos = np.array(self.node_feature[node_id, [0, 1]])
            attention_pos = np.array(self.node_feature[attention_node_index, [0, 1]])
            distance = np_vector_norm(brosis_pos - attention_pos)
            if (distance > 5.0):
                continue
            brosis_couple_node_ids_exed.append(node_id)

        # filter by Connected and length
        history_future_node_ids_exed = []
        for node_id in history_neighbour_node_ids + future_neighbour_node_ids:
            node_connected = False
            for brosis_couple_node_id in brosis_couple_node_ids_exed:
                if self.isNodeConnected(node_id, brosis_couple_node_id):
                    node_connected = True
                    break
            if node_connected:
                continue
            history_future_pos = np.array(self.node_feature[node_id, [0, 1]])
            attention_pos = np.array(self.node_feature[attention_node_index, [0, 1]])
            distance = np_vector_norm(history_future_pos - attention_pos)
            if (distance > 5.0):
                continue
            history_future_node_ids_exed.append(node_id)

        # find full seqs by key node
        all_seqs = []
        all_seqs_simplify = []
        for node_id in direct_node_ids + brosis_couple_node_ids_exed + history_future_node_ids_exed:
            forward_brosis_seqs = self.getSequence(node_id, prediction_horizon, 0)
            backwards_brosis_seqs = self.getSequence(node_id, prediction_back_horizon, 1)
            for seq_front in backwards_brosis_seqs:
                for seq_back in forward_brosis_seqs:
                    all_seqs.append(seq_front + seq_back)
                    if len(seq_front) > 0:
                        all_seqs_simplify.append([seq_front[0], seq_back[0], seq_back[-1]])
                    else:
                        all_seqs_simplify.append([seq_back[0], seq_back[0], seq_back[-1]])

        # seqs classification by destination
        all_seqs_index_list = list(range(len(all_seqs_simplify)))
        all_seqs_index_seperated = []
        while len(all_seqs_index_list) > 0:
            same_des = []
            base_index = all_seqs_index_list[0]
            same_des.append(base_index)
            for x in all_seqs_index_list[1:]:
                if self.isNodeConnected(all_seqs_simplify[base_index][-1], all_seqs_simplify[x][-1]):
                    same_des.append(x)
            all_seqs_index_seperated.append(same_des)
            for y in same_des:
                all_seqs_index_list.remove(y)

        # if same destination seqs separate or merge
        all_seqs_for_out = []
        for seq_indexs in all_seqs_index_seperated:
            mean_distance = 0.0
            if len(seq_indexs) > 1:
                for seq_index in seq_indexs[1:]:
                    mean_distance += self.distance(all_seqs_simplify[seq_index][1], all_seqs_simplify[seq_indexs[0]][1])
            mean_distance = mean_distance / len(seq_indexs)
            separate = True if mean_distance > 1.0 else False
            if separate:
                distance_2 = []
                for seq_index in seq_indexs:
                    distance_2.append(
                        self.distance(all_seqs_simplify[seq_index][1], attention_node_index) +
                        self.distance(all_seqs_simplify[seq_index][0], attention_node_index))
                all_seqs_for_out.append(all_seqs[seq_indexs[distance_2.index(min(distance_2))]])
            else:
                seq_set = set([])
                for seq_index in seq_indexs:
                    seq_set = seq_set | set(all_seqs[seq_index])
                all_seqs_for_out.append(list(seq_set))

        return all_seqs_for_out

    def findCoupleKetNodes(self, node_id, meriage_time):
        ids = []
        search_depth = 1
        node_id_point = node_id
        while search_depth < meriage_time:
            if len(self.node_pre[node_id_point]) > 1:
                break
            if len(self.node_suc[node_id_point]) == 0:
                return ids
            node_id_point = self.node_suc[node_id_point][0]
            search_depth += 1
        brosis_seqs = self.getSequence(node_id_point, search_depth, 1)
        if search_depth < meriage_time:
            for seq in brosis_seqs:
                ids.append(seq[0])
        return ids

    def findBrosisKeyNodes(self, node_id, kinship_horizon):
        ids = []
        search_depth = 1
        node_id_point = node_id
        while search_depth < kinship_horizon:
            if len(self.node_suc[node_id_point]) > 1:
                break
            if len(self.node_pre[node_id_point]) == 0:
                return ids
            node_id_point = self.node_pre[node_id_point][0]
            search_depth += 1
        brosis_seqs = self.getSequence(node_id_point, search_depth, 0)
        if search_depth < kinship_horizon:
            for seq in brosis_seqs:
                ids.append(seq[-1])
        return ids

    def findFutureNeighborKeyNodes(self, node_id, pre_horizon):
        ids = []
        if self.node_left[node_id] < 0:
            search_depth = 1
            node_id_point = copy.deepcopy(node_id)
            left_node_id = 0
            while search_depth < pre_horizon:
                if self.node_left[node_id_point] > 0:
                    left_node_id = self.node_left[node_id_point]
                    break
                if len(self.node_suc[node_id_point]) == 0:
                    return ids
                node_id_point = self.node_suc[node_id_point][0]
                search_depth += 1

            left_seqs = self.getSequence(left_node_id, search_depth, 1)
            if search_depth < pre_horizon:
                for seq in left_seqs:
                    if len(seq) > 0:
                        ids.append(seq[0])

        if self.node_right[node_id] < 0:
            search_depth = 1
            node_id_point = copy.deepcopy(node_id)
            right_node_id = 0
            while search_depth < pre_horizon:
                if self.node_right[node_id_point] > 0:
                    right_node_id = self.node_right[node_id_point]
                    break
                if len(self.node_suc[node_id_point]) == 0:
                    return ids
                node_id_point = self.node_suc[node_id_point][0]
                search_depth += 1
            right_seqs = self.getSequence(right_node_id, search_depth, 1)
            if search_depth < pre_horizon:
                for seq in right_seqs:
                    if len(seq) > 0:
                        ids.append(seq[0])
        return ids

    def findHistoryNeighborKeyNodes(self, node_id, back_horizon):
        ids = []
        if self.node_left[node_id] < 0:
            search_depth = 1
            node_id_point = copy.deepcopy(node_id)
            left_node_id = 0
            while search_depth < back_horizon:
                if self.node_left[node_id_point] > 0:
                    left_node_id = self.node_left[node_id_point]
                    break
                if len(self.node_pre[node_id_point]) == 0:
                    return ids
                node_id_point = self.node_pre[node_id_point][0]
                search_depth += 1

            left_seqs = self.getSequence(left_node_id, search_depth, 0)
            if search_depth < back_horizon:
                for seq in left_seqs:
                    if len(seq) > 0:
                        ids.append(seq[-1])
        if self.node_right[node_id] < 0:
            search_depth = 1
            node_id_point = copy.deepcopy(node_id)
            right_node_id = 0
            while search_depth < back_horizon:
                if self.node_right[node_id_point] > 0:
                    right_node_id = self.node_right[node_id_point]
                    break
                if len(self.node_pre[node_id_point]) == 0:
                    return ids
                node_id_point = self.node_pre[node_id_point][0]
                search_depth += 1
            right_seqs = self.getSequence(right_node_id, search_depth, 0)
            if search_depth < back_horizon:
                for seq in right_seqs:
                    if len(seq) > 0:
                        ids.append(seq[-1])
        return ids

    def getSequence(self, target_index, depth, direction):
        seqs = []
        path = []
        if target_index == -1:
            return seqs
        path.append(target_index)
        if direction == 0:
            self.searchForwardSeqs(depth, path, seqs)
        else:
            self.searchBackwardSeqs(depth, path, seqs)
        return seqs

    def searchForwardSeqs(self, depth, path, seqs):
        if (len(path) + 1) > depth:
            temp_path = copy.deepcopy(path)
            seqs.append(temp_path)
            return
        secs = self.node_suc[path[-1]]
        if len(secs) == 0:
            temp_path = copy.deepcopy(path)
            seqs.append(temp_path)
            return
        for sec in secs:
            path.append(sec)
            self.searchForwardSeqs(depth, path, seqs)
            path.pop()

    def searchBackwardSeqs(self, depth, path, seqs):
        if len(path) > depth:
            temp_path = copy.deepcopy(path)
            temp_path.reverse()
            temp_path.pop()
            seqs.append(temp_path)
            return
        pres = self.node_pre[path[-1]]
        if len(pres) == 0:
            temp_path = copy.deepcopy(path)
            temp_path.reverse()
            temp_path.pop()
            seqs.append(temp_path)
            return
        for pre in pres:
            path.append(pre)
            self.searchBackwardSeqs(depth, path, seqs)
            path.pop()

    def distance(self, na, nb):
        return np_vector_norm(self.node_feature[[na], [0, 1]] - self.node_feature[[nb], [0, 1]])


def getHeadingDiff(heading_a, heading_b):
    heading_a_wrap = heading_a % (math.pi * 2.0)
    if heading_a_wrap < 0.0:
        heading_a_wrap += math.pi * 2.0
    heading_b_wrap = heading_b % (math.pi * 2.0)
    if heading_b_wrap < 0.0:
        heading_b_wrap += math.pi * 2.0
    heading_diff = math.fabs(heading_a_wrap - heading_b_wrap)
    if heading_diff > math.pi:
        heading_diff = math.pi * 2.0 - heading_diff
    return heading_diff
