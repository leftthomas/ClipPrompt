import numpy as np
from scipy.spatial.distance import cdist


def sake_metric(predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, k=None):
    if k is None:
        k = {'precision': 100, 'map': predicted_features_gallery.shape[0]}
    if k['precision'] is None:
        k['precision'] = 100
    if k['map'] is None:
        k['map'] = predicted_features_gallery.shape[0]

    scores = -cdist(predicted_features_query, predicted_features_gallery, metric='cosine')
    gt_labels_query = gt_labels_query.flatten()
    gt_labels_gallery = gt_labels_gallery.flatten()
    aps = map_sake(gt_labels_gallery, predicted_features_query, gt_labels_query, scores, k=k['map'])
    prec = prec_sake(gt_labels_gallery, predicted_features_query, gt_labels_query, scores, k=k['precision'])
    return sum(aps) / len(aps), prec


def map_sake(gt_labels_gallery, predicted_features_query, gt_labels_query, scores, k=None):
    mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    mean_mAP = []
    for fi in range(predicted_features_query.shape[0]):
        mapi = eval_ap(gt_labels_query[fi], scores[fi], gt_labels_gallery, top=k)
        mAP_ls[gt_labels_query[fi]].append(mapi)
        mean_mAP.append(mapi)
    return mean_mAP


def prec_sake(gt_labels_gallery, predicted_features_query, gt_labels_query, scores, k=None):
    # compute precision for two modalities
    prec_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    mean_prec = []
    for fi in range(predicted_features_query.shape[0]):
        prec = eval_precision(gt_labels_query[fi], scores[fi], gt_labels_gallery, top=k)
        prec_ls[gt_labels_query[fi]].append(prec)
        mean_prec.append(prec)
    return np.nanmean(mean_prec)


def eval_ap(inst_id, scores, gt_labels, top=None):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]
    tot_pos = np.sum(pos_flag)

    sort_idx = np.argsort(-scores)
    tp = pos_flag[sort_idx]
    fp = np.logical_not(tp)

    if top is not None:
        top = min(top, tot)
        tp = tp[:top]
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        prec = tp / (tp + fp)
    except:
        return np.nan

    ap = voc_ap(rec, prec)
    return ap


def voc_ap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre) - 2, -1, -1):
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap


def eval_precision(inst_id, scores, gt_labels, top=100):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]
    top = min(top, tot)
    sort_idx = np.argsort(-scores)
    return np.sum(pos_flag[sort_idx][:top]) / top
