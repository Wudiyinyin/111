# Copyright (c) 2021 Li Auto Company. All rights reserved.
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from datasets.database.sample_database import SampleDataset

parser = argparse.ArgumentParser(description='train intention network.')

parser.add_argument('-db', '--database_path', type=str, default='', required=True, help='database path')

parser.add_argument('-type', '--database_type', type=str, default='lmdb', help='database type')

parser.add_argument('-f', '--sample_list_file', type=str, default=None, help='sample list file')


def plot_vector(start, end, color, width, alpha, headwidth, headlength):
    '''
    plot an array of vector as arrow
    '''

    delta = end - start
    plt.quiver(start[:, 0],
               start[:, 1],
               delta[:, 0],
               delta[:, 1],
               color=color,
               width=width,
               headwidth=headwidth,
               headlength=headlength,
               headaxislength=headlength,
               alpha=alpha,
               units='xy',
               angles='xy',
               scale_units='xy',
               scale=1.0)


def plot_map_points(sample):
    print('all point num: ', len(sample.map_points))
    x = [point.x for point in sample.map_points]
    y = [point.y for point in sample.map_points]
    plt.scatter(x, y, s=0.03)


def plot_sample(
    sample,
    agent_id=-1,
    use_global_map=False,
    show_agent=True,
    show_obstacle=True,
    show_lane_node=True,
    show_lane_id=False,
    show_lane_relation=False,
    show_boundary=True,
    show_polygon=True,
    show_stopline=True,
    show_intention=False,
):

    agent_polyline_ids = None
    if not use_global_map:  # use local map
        agent_polyline_ids = {
            'Obstacle': [],
            'Lane': [],
            'LaneBoundary': [],
            'RoadBoundary': [],
            'Polygon': [],
            'Stopline': []
        }
        for obstacle in sample.obstacle_features:
            if obstacle.id == agent_id:
                # print(f"obstacle {obstacle.id} has {len(obstacle.polyline_info)} polyline_info")
                for polyline in obstacle.polyline_info:
                    # print(f'  polyline_id: {polyline.polyline_id}  polyline_type: {polyline.type}')
                    if polyline.type == polyline.Obstacle:
                        agent_polyline_ids['Obstacle'].append(int(polyline.polyline_id))
                    elif polyline.type == polyline.Lane:
                        agent_polyline_ids['Lane'].append(polyline.polyline_id)
                    elif polyline.type == polyline.LaneBoundary:
                        agent_polyline_ids['LaneBoundary'].append(polyline.polyline_id)
                    elif polyline.type == polyline.RoadBoundary:
                        agent_polyline_ids['RoadBoundary'].append(polyline.polyline_id)
                    elif polyline.type == polyline.Polygon:
                        agent_polyline_ids['Polygon'].append(polyline.polyline_id)
                    elif polyline.type == polyline.Stopline:
                        agent_polyline_ids['Stopline'].append(polyline.polyline_id)
                # print(agent_polyline_ids)

    # obstacle
    if show_obstacle:
        print("obstacle_features", len(sample.obstacle_features))
        for obstacle in sample.obstacle_features:
            print(f"obs_id: {obstacle.id} obs_vec:{len(obstacle.obs_vec)}")
            if len(obstacle.obs_vec) == 0:
                continue

            if (obstacle.id != agent_id) and agent_polyline_ids and not (obstacle.id in agent_polyline_ids['Obstacle']):
                continue

            obstacle_start = []
            obstacle_end = []
            for frame in obstacle.obs_vec:
                obstacle_start.append([frame.vec.points[0].x, frame.vec.points[0].y])
                obstacle_end.append([frame.vec.points[1].x, frame.vec.points[1].y])
            obstacle_start = np.array(obstacle_start)
            obstacle_end = np.array(obstacle_end)

            if show_agent and obstacle.id == agent_id:
                plot_vector(obstacle_start, obstacle_end, 'red', 0.2, 1.0, 1.5, 1.5)
                # print(f'agent pos: {obstacle_start[0][0]}, {obstacle_start[0][1]}')
            else:
                plot_vector(obstacle_start, obstacle_end, 'blue', 0.1, 1.0, 1.5, 1.5)
                # print(f'obs pos: {obstacle_start[0][0]}, {obstacle_start[0][1]}')

            if len(obstacle.obs_future_vec) > 1:
                obstacle_start = []
                obstacle_end = []
                for frame in obstacle.obs_future_vec:
                    obstacle_start.append([frame.vec.points[0].x, frame.vec.points[0].y])
                    obstacle_end.append([frame.vec.points[1].x, frame.vec.points[1].y])
                obstacle_start = np.array(obstacle_start)
                obstacle_end = np.array(obstacle_end)
                plot_vector(obstacle_start, obstacle_end, 'green', 0.1, 1.0, 1.5, 1.5)

    # if show_intention:
    #     intention_start = node_start[[sample.intention]]
    #     intention_end = node_end[[sample.intention]]
    #     plot_vector(intention_start, intention_end, 'green', 0.3, 1.0, 1.5, 0.7)

    # lanes
    if show_lane_node:
        print("lanes", len(sample.map_polyline.lanes))
        for lane in sample.map_polyline.lanes:
            if len(lane.lane_vector) == 0:
                continue

            if agent_polyline_ids and not (lane.id in agent_polyline_ids['Lane']):
                continue

            vector_start = []
            vector_end = []
            for node in lane.lane_vector:
                vector_start.append([node.vec.points[0].x, node.vec.points[0].y])
                vector_end.append([node.vec.points[1].x, node.vec.points[1].y])
                if show_lane_id:
                    plt.text((node.vec.points[0].x + node.vec.points[1].x) / 2,
                             (node.vec.points[0].y + node.vec.points[1].y) / 2, str(lane.id))
            vector_start = np.array(vector_start)
            vector_end = np.array(vector_end)
            # node_color = [['gray', 'yellow'][int(
            #     node.attribute.has_traffic_control)] for node in sample.lane_graph.lane_nodes]
            plot_vector(vector_start, vector_end, 'gray', 0.3, 0.5, 1.5, 0.7)

    # lane_boundaries
    if show_boundary:
        # print("lane_boundarys", len(sample.map_polyline.lane_boundarys))
        for lane in sample.map_polyline.lane_boundarys:
            # print("lane_boundary_vec", len(lane.lane_boundary_vec))
            if len(lane.lane_boundary_vec) == 0:
                continue

            if agent_polyline_ids and not (lane.id in agent_polyline_ids['LaneBoundary']):
                # print(f"lane_id {lane.id} not in local LaneBoundary {agent_polyline_ids['LaneBoundary']}")
                continue

            vector_start = []
            vector_end = []
            for node in lane.lane_boundary_vec:
                vector_start.append([node.vec.points[0].x, node.vec.points[0].y])
                vector_end.append([node.vec.points[1].x, node.vec.points[1].y])
            vector_start = np.array(vector_start)
            vector_end = np.array(vector_end)
            plot_vector(vector_start, vector_end, 'gray', 0.1, 0.5, 1.5, 1.5)

    # road_boundaries
    if show_boundary:
        # print("road_boundaries", len(sample.map_polyline.road_boundarys))
        for lane in sample.map_polyline.road_boundarys:
            # print("road_boundary_vec", len(lane.road_boundary_vec))
            if len(lane.road_boundary_vec) == 0:
                continue

            if agent_polyline_ids and not (lane.id in agent_polyline_ids['RoadBoundary']):
                # print(f"lane_id {lane.id} not in local RoadBoundary {agent_polyline_ids['RoadBoundary']}")
                continue

            vector_start = []
            vector_end = []
            for node in lane.road_boundary_vec:
                vector_start.append([node.vec.points[0].x, node.vec.points[0].y])
                vector_end.append([node.vec.points[1].x, node.vec.points[1].y])
            vector_start = np.array(vector_start)
            vector_end = np.array(vector_end)
            plot_vector(vector_start, vector_end, 'black', 0.1, 1.0, 1.5, 1.5)

    # polygon_polylines
    if show_polygon:
        print("polygon_polylines", len(sample.map_polyline.map_polylines))
        for lane in sample.map_polyline.map_polylines:
            print("polygon_vec", len(lane.polygon_vec))
            if len(lane.polygon_vec) == 0:
                continue

            if agent_polyline_ids and not (lane.id in agent_polyline_ids['Polygon']):
                continue

            vector_start = []
            vector_end = []
            for node in lane.polygon_vec:
                vector_start.append([node.vec.points[0].x, node.vec.points[0].y])
                vector_end.append([node.vec.points[1].x, node.vec.points[1].y])
            vector_start = np.array(vector_start)
            vector_end = np.array(vector_end)
            plot_vector(vector_start, vector_end, 'gray', 0.1, 1.0, 1.5, 1.5)

    if show_stopline:
        # print("stop_polylines", len(sample.dynamic_polyline.stoplines))
        for lane in sample.dynamic_polyline.stoplines:
            # print("stopline_vec", len(lane.stopline_vec))
            if len(lane.stopline_vec) == 0:
                continue

            if agent_polyline_ids and not (lane.id in agent_polyline_ids['Stopline']):
                continue

            vector_start = []
            vector_end = []
            for node in lane.stopline_vec:
                vector_start.append([node.vec.points[0].x, node.vec.points[0].y])
                vector_end.append([node.vec.points[1].x, node.vec.points[1].y])
            vector_start = np.array(vector_start)
            vector_end = np.array(vector_end)
            plot_vector(vector_start, vector_end, 'red', 0.1, 1.0, 1.5, 1.5)

    # polyline relation
    if show_lane_relation:
        # generate lane info
        lane_id_dict = {}
        for lane in sample.map_polyline.lanes:
            if len(lane.lane_vector) == 0:
                continue

            if agent_polyline_ids and not (lane.id in agent_polyline_ids['Lane']):
                continue

            lane_vec_start = lane.lane_vector[0]
            lane_vec_end = lane.lane_vector[-1]
            lane_start = [lane_vec_start.vec.points[0].x, lane_vec_start.vec.points[0].y]
            lane_end = [lane_vec_end.vec.points[1].x, lane_vec_end.vec.points[1].y]
            lane_id_dict[lane.id] = {
                'start': lane_start,
                'end': lane_end,
                'middle': [(lane_start[0] + lane_end[0]) / 2, (lane_start[1] + lane_end[1]) / 2]
            }

        # show whole polyline
        polyline_start = []
        polyline_end = []
        for lane in sample.map_polyline.lanes:
            if len(lane.lane_vector) == 0:
                continue

            if agent_polyline_ids and not (lane.id in agent_polyline_ids['Lane']):
                continue

            lane_id = lane.id
            polyline_start.append(lane_id_dict[lane_id]['start'])
            polyline_end.append(lane_id_dict[lane_id]['end'])
            print(f'lane_id {lane.id} total_length {lane.total_length}')

        polyline_start = np.array(polyline_start)
        polyline_end = np.array(polyline_end)
        plot_vector(polyline_start, polyline_end, 'black', 0.2, 0.5, 1.5, 0.7)

        # show pre edge
        edge_pre_start = []
        edge_pre_end = []
        for lane in sample.map_polyline.lanes:
            if len(lane.lane_vector) == 0:
                continue

            if agent_polyline_ids and not (lane.id in agent_polyline_ids['Lane']):
                continue

            print("lane.predecessor", len(lane.predecessor))
            if len(lane.predecessor) == 0:
                continue
            lane_id = lane.id
            for pre_id in lane.predecessor:
                edge_pre_start.append(lane_id_dict[pre_id]['middle'])
                edge_pre_end.append(lane_id_dict[lane_id]['middle'])
        edge_pre_start = np.array(edge_pre_start)
        edge_pre_end = np.array(edge_pre_end)
        assert edge_pre_start.shape == edge_pre_end.shape
        if edge_pre_start.shape[0] > 0:
            plot_vector(edge_pre_start, edge_pre_end, 'white', 0.05, 1.0, 1.5, 1.5)

        # show suc edge
        edge_suc_start = []
        edge_suc_end = []
        for lane in sample.map_polyline.lanes:
            if len(lane.lane_vector) == 0:
                continue

            if agent_polyline_ids and not (lane.id in agent_polyline_ids['Lane']):
                continue

            print("lane.successor", len(lane.successor))
            if len(lane.successor) == 0:
                continue
            lane_id = lane.id
            for suc_id in lane.successor:
                edge_suc_start.append(lane_id_dict[suc_id]['middle'])
                edge_suc_end.append(lane_id_dict[lane_id]['middle'])
        edge_suc_start = np.array(edge_suc_start)
        edge_suc_end = np.array(edge_suc_end)
        assert edge_suc_start.shape == edge_suc_end.shape
        if edge_suc_start.shape[0] > 0:
            plot_vector(edge_suc_start, edge_suc_end, 'black', 0.015, 1.0, 1.5, 1.5)

        # show left edge
        edge_left_start = []
        edge_left_end = []
        for lane in sample.map_polyline.lanes:
            if len(lane.lane_vector) == 0:
                continue

            if agent_polyline_ids and not (lane.id in agent_polyline_ids['Lane']):
                continue

            print("lane.left_neighbor", len(lane.left_neighbor))
            if len(lane.left_neighbor) == 0:
                continue
            lane_id = lane.id
            for left_id in lane.left_neighbor:
                edge_left_start.append(lane_id_dict[left_id]['middle'])
                edge_left_end.append(lane_id_dict[lane_id]['middle'])
        edge_left_start = np.array(edge_left_start)
        edge_left_end = np.array(edge_left_end)
        assert edge_left_start.shape == edge_left_end.shape
        if edge_left_start.shape[0] > 0:
            plot_vector(edge_left_start, edge_left_end, 'red', 0.02, 1.0, 1.5, 1.5)

        # show right edge
        edge_right_start = []
        edge_right_end = []
        for lane in sample.map_polyline.lanes:
            if len(lane.lane_vector) == 0:
                continue

            if agent_polyline_ids and not (lane.id in agent_polyline_ids['Lane']):
                continue

            print("lane.right_neighbor", len(lane.right_neighbor))
            if len(lane.right_neighbor) == 0:
                continue
            lane_id = lane.id
            for right_id in lane.right_neighbor:
                edge_right_start.append(lane_id_dict[right_id]['middle'])
                edge_right_end.append(lane_id_dict[lane_id]['middle'])
        edge_right_start = np.array(edge_right_start)
        edge_right_end = np.array(edge_right_end)
        assert edge_right_start.shape == edge_right_end.shape
        if edge_right_start.shape[0] > 0:
            plot_vector(edge_right_start, edge_right_end, 'green', 0.007, 1.0, 1.5, 1.5)


def show_sample(figure_title, agent_id, use_global_map, use_map_points, sample):
    plt.figure(figure_title)
    if use_map_points:
        plt.subplot(1, 2, 1)
    plt.title(figure_title)
    # plt.grid()
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plot_sample(sample, agent_id, use_global_map)
    if use_map_points:
        plt.subplot(1, 2, 2)
        plot_map_points(sample)

    figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    figManager.resize(*figManager.window.maxsize())
    plt.show()
    # plt.savefig(f'{output_dir}/{figure_title}.svg')


if __name__ == "__main__":
    args = parser.parse_args()

    dataset = SampleDataset(args.database_path, args.database_type, args.sample_list_file)
    print(f'Dataset sample number: {len(dataset)}')

    scene_id_set = set()
    only_plot_first = False
    is_plot_local_map = True
    for i in range(len(dataset)):
        sample = dataset[i]
        # plot adc with global map
        agent_id = -1
        use_global_map = True
        use_map_points = True
        if only_plot_first and sample.scene_id in scene_id_set:
            continue
        figure_title = "adc with global map " + sample.scene_id + "_" + str(sample.seq_num) + "_" + str(agent_id)
        show_sample(figure_title, agent_id, use_global_map, use_map_points, sample)
        scene_id_set.add(sample.scene_id)

        # plot each obstacle
        if not is_plot_local_map:
            continue
        for obstacle in sample.obstacle_features:
            agent_id = obstacle.id
            use_global_map = False
            figure_title = sample.scene_id + "_" + str(sample.seq_num) + "_" + str(agent_id)
            show_sample(figure_title, agent_id, use_global_map, use_map_points, sample)
