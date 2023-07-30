import numpy as np
import pandas as pd

def calc_iou(mask_a, mask_b):
    intersection = (mask_a + mask_b >= 2).astype(np.float32).sum()
    iou = intersection / (mask_a + mask_b >= 1).astype(np.float32).sum()
    return iou

def match(matrix, iou_thread, img_name=False):
    matched_gts = np.arange(matrix.shape[0])
    matched_ranks = matrix.argsort()[:, -1]
    for i, j in zip(matched_gts, matched_ranks):
        if matrix[i][j] < iou_thread:
            matched_ranks[i] = -1
    if len(set(matched_ranks[matched_ranks != -1])) < len(matched_ranks[matched_ranks != -1]):
        for i in set(matched_ranks):
            if i >= 0:
                index_i = np.nonzero(matched_ranks == i)[0]
                if len(index_i) > 1:
                    score_index = matched_ranks[index_i[0]]
                    ious = matrix[:, score_index][index_i]
                    max_index = index_i[ious.argsort()[-1]]
                    rm_index = index_i[np.nonzero(index_i != max_index)[0]]
                    matched_ranks[rm_index] = -1
    if len(set(matched_ranks[matched_ranks != -1])) < len(matched_ranks[matched_ranks != -1]):
        print('match Error')
        raise KeyError
    if len(matched_ranks) < matrix.shape[1]:
        for i in range(matrix.shape[1]):
            if i not in matched_ranks:
                matched_ranks = np.append(matched_ranks, i)
    return matched_ranks


def get_rank_index(gt_masks, segmaps, iou_thread, rank_scores):
    ious = np.zeros([len(gt_masks), len(segmaps)])
    for i in range(len(gt_masks)):
        for j in range(len(segmaps)):
            ious[i][j] = calc_iou(gt_masks[i], segmaps[j])
    matched_ranks = match(ious, iou_thread)
    unmatched_index = np.argwhere(matched_ranks == -1).squeeze(1)
    matched_ranks = matched_ranks[matched_ranks >= 0]
    unmatch_rank = rank_scores[[i for i in range(len(rank_scores)) if i not in matched_ranks]] + 1
    rank_vis = rank_scores[matched_ranks] + 1
    rank_scores = rank_scores[matched_ranks]
    rank_index = np.array([sorted(rank_scores).index(a) + 1 for a in rank_scores])
    for i in range(len(unmatched_index)):
        rank_index = np.insert(rank_index, unmatched_index[i], 0)
        rank_vis = np.insert(rank_vis, unmatched_index[i], 0)
    rank_index = rank_index[:len(gt_masks)]
    l = len(rank_vis) - len(gt_masks)
    if l > 0:
        rank_vis[-l:] = 0
    rank_vis = np.append(rank_vis, unmatch_rank)
    
    return rank_index, rank_vis

def evalu_org(results, iou_thread):
    print('\nCalculating Sprman ...\n')
    p_sum = 0
    num = len(results)

    for indx, result in enumerate(results):
        print('\r{}/{}'.format(indx+1, len(results)), end="", flush=True)
        gt_masks = result['gt_masks']
        segmaps = result['segmaps']
        gt_ranks = result['gt_ranks']
        rank_scores = result['rank_scores']
        rank_scores = np.array(rank_scores)[:, None]

        if len(gt_ranks) == 1:
            num = num - 1
            continue

        gt_index = np.array([sorted(gt_ranks).index(a) + 1 for a in gt_ranks])

        if len(segmaps) == 0:
            rank_index = np.zeros_like(gt_ranks)
        else:
            rank_index = get_rank_index(gt_masks, segmaps, iou_thread, rank_scores)

        gt_index = pd.Series(gt_index)
        rank_index = pd.Series(rank_index)
        if rank_index.var() == 0:
            p = 0
        else:
            p = gt_index.corr(rank_index, method='pearson')
        if not np.isnan(p):
            p_sum += p
        else:
            num -= 1

    fianl_p = p_sum/num
    return fianl_p

def evalu(gt_masks, gt_ranks, pred_mask, iou_thread):
    gt_masks = np.array(gt_masks)
    gt_ranks = np.array(gt_ranks)

    segmaps = np.array(pred_mask.pred_masks)
    rank_scores = np.array(pred_mask.pred_ranks)

    if len(gt_ranks) == 1:
        return -2, []

    gt_index = np.array([sorted(gt_ranks).index(a) + 1 for a in gt_ranks])

    if len(segmaps) == 0:
        rank_index = np.zeros_like(gt_ranks)
        rank_vis = np.zeros_like(gt_ranks)
    else:
        rank_index, rank_vis = get_rank_index(gt_masks, segmaps, iou_thread, rank_scores)

    gt_index = pd.Series(gt_index)
    rank_index = pd.Series(rank_index)
    if rank_index.var() == 0:
        p = 0
    else:
        p = gt_index.corr(rank_index, method='pearson')
    if np.isnan(p):
        return -2, []

    return p, rank_vis