# Copyright (c) 2021 Li Auto Company. All rights reserved.
import sys

import matplotlib.pyplot as plt
import numpy as np
from datasets.database.sample_database import SampleDataset


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


def plot_sample(sample,
                show_lane_node=True,
                show_lane_edge=False,
                show_agent=True,
                show_obstacle=False,
                show_boundary=False,
                show_attention=False,
                show_intention=False,
                show_component_id=False):

    # boundaries
    if show_boundary:
        for boundary in sample.drivable_boundaries:
            x = [point.x for point in boundary.points]
            y = [point.y for point in boundary.points]
            pos = np.array([x, y]).transpose(1, 0)
            plot_vector(pos[:-1], pos[1:], 'black', 0.1, 1.0, 1.5, 1.5)

    # lanegraph node
    if show_lane_node:
        node_start = np.array([[node.attribute.start.x, node.attribute.start.y] for node in sample.lane_graph.lane_nodes
                              ])
        node_end = np.array([[node.attribute.end.x, node.attribute.end.y] for node in sample.lane_graph.lane_nodes])
        # node_color = [['gray', 'yellow'][int(
        #     node.attribute.has_traffic_control)] for node in sample.lane_graph.lane_nodes]
        node_color = [['gray', 'yellow'][0] for node in sample.lane_graph.lane_nodes]
        plot_vector(node_start, node_end, node_color, 0.3, 0.5, 1.5, 0.7)

        node_pos = (node_start + node_end) * 0.5

    # lanegraph edge
    if show_lane_edge:
        edge_pre = []
        for node_id, node in enumerate(sample.lane_graph.lane_nodes):
            for pre_id in node.predecessors:
                edge_pre.append([pre_id, node_id])
        edge_pre = np.array(edge_pre, dtype=np.long)
        plot_vector(node_pos[edge_pre[:, 0]], node_pos[edge_pre[:, 1]], 'white', 0.05, 1.0, 1.5, 1.5)

        edge_suc = []
        for node_id, node in enumerate(sample.lane_graph.lane_nodes):
            for suc_id in node.successors:
                edge_suc.append([suc_id, node_id])
        edge_suc = np.array(edge_suc, dtype=np.long)
        plot_vector(node_pos[edge_suc[:, 0]], node_pos[edge_suc[:, 1]], 'black', 0.015, 1.0, 1.5, 1.5)

        edge_left = []
        for node_id, node in enumerate(sample.lane_graph.lane_nodes):
            left_id = node.left_neighbour
            if left_id != -1:
                edge_left.append([left_id, node_id])
        edge_left = np.array(edge_left, dtype=np.long)
        plot_vector(node_pos[edge_left[:, 0]], node_pos[edge_left[:, 1]], 'red', 0.02, 1.0, 1.5, 1.5)

        edge_right = []
        for node_id, node in enumerate(sample.lane_graph.lane_nodes):
            right_id = node.right_neighbour
            if right_id != -1:
                edge_right.append([right_id, node_id])
        edge_right = np.array(edge_right, dtype=np.long)
        plot_vector(node_pos[edge_right[:, 0]], node_pos[edge_right[:, 1]], 'green', 0.007, 1.0, 1.5, 1.5)

    # plot obstacles
    if show_obstacle:
        for actor in sample.trajectories[1:]:
            x = [state.pos.x for state in actor.states]
            y = [state.pos.y for state in actor.states]
            pos = np.array([x, y]).transpose(1, 0)
            plot_vector(pos[:-1], pos[1:], 'blue', 0.1, 1.0, 1.5, 1.5)

    # plot the agent
    if show_agent:
        x = [state.pos.x for state in sample.trajectories[0].states]
        y = [state.pos.y for state in sample.trajectories[0].states]
        pos = np.array([x, y]).transpose(1, 0)
        history_pos = pos[:20]
        future_pos = pos[19:]
        plot_vector(history_pos[:-1], history_pos[1:], 'red', 0.1, 1.0, 1.5, 1.5)
        plot_vector(future_pos[:-1], future_pos[1:], 'green', 0.1, 1.0, 1.5, 1.5)

    # plot component marker
    if show_component_id:
        for node in sample.lane_graph.lane_nodes:
            plt.text((node.attribute.start.x + node.attribute.end.x) / 2,
                     (node.attribute.start.y + node.attribute.end.y) / 2, str(node.father_component))

    # plot attention
    if show_attention:
        attention_start = node_start[[sample.attention]]
        attention_end = node_end[[sample.attention]]
        plot_vector(attention_start, attention_end, 'red', 0.3, 1.0, 1.5, 0.7)

    # plot intention
    if show_intention:
        intention_start = node_start[[sample.intention]]
        intention_end = node_end[[sample.intention]]
        plot_vector(intention_start, intention_end, 'green', 0.3, 1.0, 1.5, 0.7)


if __name__ == "__main__":
    database_path = sys.argv[1]
    sample_list_file = None
    if len(sys.argv) >= 3:
        sample_list_file = sys.argv[2]

    dataset = SampleDataset(database_path, 'lmdb', sample_list_file)
    print(f'Dataset sample number: {len(dataset)}')

    for i in range(len(dataset)):
        plot_sample(dataset[i])
        plt.axis('equal')
        figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        figManager.resize(*figManager.window.maxsize())
        plt.show()
