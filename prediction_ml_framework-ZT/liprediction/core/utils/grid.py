# Copyright (c) 2021 Li Auto Company. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import torch


def pos_to_cord(pos, grid_reso, grid_size):
    '''
                x                                      x(row)
                |                             ---|--|--|
                |                                |  |  |
         y <----------                        ---------|
                |                                |  |  |
                |                      y(col) <---------
        center_coord_true_reso  ->  corner_coord_grid_reso

        pos: n-dim array [N, 2]
        return: n-dim array [N, 2]
    '''

    assert isinstance(pos, np.ndarray)
    # center_coord_true_reso -> corner_coord_true_reso -> corner_coord_grid_reso
    cord = np.floor((pos + grid_size * grid_reso / 2) / grid_reso)
    return cord.astype(np.int64)


def cord_to_pos(cord, grid_reso, grid_size):
    '''
        corner_coord_grid_reso  ->  abscenter_coord_grid_reso  -> center_coord_true_reso

        cord: n-dim array [N, 2]
        return: n-dim array [N, 2]
    '''

    assert isinstance(cord, np.ndarray)
    # corner_coord_grid_reso  ->  abscenter_coord_grid_reso  -> center_coord_true_reso
    pos = (cord - (grid_size - 1) / 2) * grid_reso
    return pos.astype(np.float32)


def cord_to_index(cord, grid_size):
    '''
        corner_coord_grid_reso -> index

        cord: n-dim array [N, 2] = [[x,y],...] or list [2] = [x,y]
        return: n-dim array [N] or int num
    '''
    assert isinstance(cord, np.ndarray)
    assert cord.ndim in [1, 2]
    if cord.ndim == 1:  # [2,] = [1, 2]
        # [2,] -> [1, 2]
        cord = np.expand_dims(cord, axis=0)
    assert cord.ndim == 2

    is_over_border = np.any(np.where((cord < 0) | (cord > grid_size - 1), True, False), axis=1)

    cord = np.clip(cord, 0, grid_size - 1)  # in case of cord exceed dim
    x = cord[:, 0]
    y = cord[:, 1]
    # [N] + [N] -> [N]
    return x + y * grid_size, is_over_border


def index_to_cord(index, grid_size):
    '''
        index -> corner_coord_grid_reso

        index: n-dim array [N] or int num
        return: n-dim array [N, 2] = [[x,y],...] or [2] = [x,y]
    '''
    assert isinstance(index, np.ndarray)
    assert index.ndim == 1
    # cord_x = np.remainder(index, dim)
    # cord_y = np.floor_divide(index, dim)
    #  % operator can be used as a shorthand for np.remainder on ndarrays.
    # // operator can be used as a shorthand for np.floor_divide on ndarrays
    cord_x = index % grid_size
    cord_y = index // grid_size
    # [N] + [N] -> [N, 2]
    return np.stack((cord_x, cord_y), axis=1)


def pos_to_index(pos, grid_reso, grid_size):
    return cord_to_index(pos_to_cord(pos, grid_reso, grid_size), grid_size)


def index_to_pos(index, grid_reso, grid_size):
    return cord_to_pos(index_to_cord(index, grid_size), grid_reso, grid_size)


def index_to_pos_tensor(grid_idx, grid_center_offset, grid_reso, grid_size):
    '''
        grid_idx: [B, candi_num]
        grid_center_offset: [B, 2, candi_num]
    '''

    # index_to_cord: index -> corner_coord_grid_reso
    # [B, candi_num]
    cord_x = torch.remainder(grid_idx, grid_size)
    # [B, candi_num]
    cord_y = torch.floor_divide(grid_idx, grid_size)
    # [B, candi_num],[B, candi_num] -> [B, candi_num, 2]
    grid_cord = torch.stack([cord_x, cord_y], dim=-1)

    # cord_to_pos: corner_coord_grid_reso  ->  abscenter_coord_grid_reso  -> center_coord_true_reso
    # [B, candi_num, 2] -> [B, candi_num, 2]
    grid_center = (grid_cord - (grid_size - 1) / 2) * grid_reso
    # [B, candi_num, 2] + [B, 2, candi_num]
    return grid_center + grid_center_offset


def gen_grid_center_embeding(batch_size, dim, device='cpu'):
    # corner_coord -> center_coord:
    #   [0,1,2,...,d-1] - ((d-1)/2) -> [-(d-1)/2, ..., +(d-1)/2]
    grid_center_cord = torch.arange(dim, device=device) - (dim - 1) / 2

    # center_coord -> normalized_coord:
    #   [-(d-1)/2, ..., +(d-1)/2] ->"*1/(d/2)"-> [-1, ..., +1]
    norm_center_cord = grid_center_cord / (dim / 2)

    # [d] -> [1, d] -> [d, d]  ([-, ..., +])
    cord_x_embeding = norm_center_cord.expand(dim, dim)  # cord.unsqueeze(0).expand(d, d)
    # [d] -> [d, 1] -> [d, d] ([-, ..., +]^T)
    cord_y_embeding = norm_center_cord.unsqueeze(1).expand(dim, dim)

    # [[d,d],[d,d]] -> [d,d,2] -> [b,d,d,2]
    pix_embeding = torch.stack([cord_x_embeding, cord_y_embeding], dim=-1)
    # [d,d,2] -> [b,d,d,2] -> [b, d*d, 2]
    batch_pix_embeding = pix_embeding.expand(batch_size, dim, dim, 2).reshape(batch_size, -1, 2)
    return batch_pix_embeding


def gen_grid_center_pos(reso, dim):
    # corner_coord -> center_coord:
    # [0,...,d-1] - (d-1)/2 -> [-(d-1)/2, ..., +(d-1)/2], grid_center=(d-1)/2
    grid_center_cord = np.float32(np.arange(dim) - (dim - 1) / 2)

    # center_coord -> normalized_coord:
    #   [-(d-1)/2, ..., +(d-1)/2] ->"*reso" -> [-N, ..., +N]
    true_center_cord = grid_center_cord * reso

    # [d] -> [1, d] -> [d, d]  ([-, ..., +])
    cord_x = np.broadcast_to(np.expand_dims(true_center_cord, 0), (dim, dim))
    # [d] -> [d, 1] -> [d, d] ([-, ..., +]^T)
    cord_y = np.broadcast_to(np.expand_dims(true_center_cord, 1), (dim, dim))

    # [[d,d],[d,d]] -> [d,d,2]
    cord_xy_matrix = np.stack([cord_x, cord_y], axis=-1)
    # [d,d,2] -> [d*d, 2]
    # cord_xy_array = cord_xy_matrix.reshape(-1, 2)
    return true_center_cord, true_center_cord, cord_xy_matrix


if __name__ == '__main__':
    # gt_grid_idx = cord_to_index(np.array([[-2, 3], [2, 5], [3, 1]]), 4)
    grid_reso = 1.0
    grid_size = 4
    gt_pos = np.array([[1.2, 0.6], [-1.3, -1.8], [1.3, -1.8]], dtype=float)
    print("gt_pos\n", gt_pos)

    gt_grid_cord = pos_to_cord(gt_pos, grid_reso, grid_size)
    gt_grid_idx, gt_grid_over_border = cord_to_index(gt_grid_cord, grid_size)
    print("gt_grid_cord\n", gt_grid_cord)
    print("gt_grid_idx\n", gt_grid_idx)
    print("gt_grid_idx.shape\n", gt_grid_idx.shape)

    gt_grid_cord_1 = index_to_cord(gt_grid_idx, grid_size)
    gt_grid_center = cord_to_pos(gt_grid_cord_1, grid_reso, grid_size)
    print("gt_grid_cord_1\n", gt_grid_cord_1)
    print("gt_grid_center\n", gt_grid_center)

    gt_grid_offset = gt_pos - gt_grid_center
    print("gt_pos\n", gt_pos)
    print("gt_grid_center\n", gt_grid_center)
    print("gt_grid_offset\n", gt_grid_offset)

    true_center_cord_x, true_center_cord_y, cord_xy_matrix = gen_grid_center_pos(grid_reso, grid_size)
    print("true_center_cord_x\n", true_center_cord_x)
    print("true_center_cord_y\n", true_center_cord_y)
    print("cord_xy_matrix\n", cord_xy_matrix)

    batch_pix_embeding = gen_grid_center_embeding(1, grid_size, device='cpu')
    print("batch_pix_embeding\n", batch_pix_embeding)

    dim = grid_size
    # heatmap = torch.zeros(dim * dim)
    # heatmap[11] = 1.0
    # heatmap = heatmap.view(dim, dim)
    heatmap = cord_xy_matrix.sum(axis=-1)
    print("headmap\n", heatmap)
    # Wistia, YlOrBr, Oranges, YlOrBr, YlOrRd, YlGnBu, YlGn, hot
    # ref: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    plt.pcolormesh(true_center_cord_x, true_center_cord_y, heatmap, cmap='Wistia', alpha=1.0, shading='auto')
    plt.xlabel('x')
    plt.ylabel('y')

    # show coord and index
    for i in range(cord_xy_matrix.shape[0]):
        for j in range(cord_xy_matrix.shape[1]):
            x = cord_xy_matrix[i, j][0]
            y = cord_xy_matrix[i, j][1]
            plt.text(x, y, '(%.4f,%.4f)' % (x, y), horizontalalignment='center', verticalalignment='center')

    figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    figManager.resize(*figManager.window.maxsize())
    plt.show()
