import torch
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def evaluate_model(
    model,
    loader,
    loss_fn,
    device
):

    model.eval()

    total_loss = 0

    precision_list = []
    recall_list = []
    f1_list = []
    iou_list = []

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            raw_outputs = model(images)
            outputs = raw_outputs['out'] if isinstance(raw_outputs, dict) else raw_outputs

            loss = loss_fn(
                outputs,
                masks
            )

            total_loss += loss.item()

            precision, recall, f1, iou = compute_metrics(
                outputs,
                masks
            )

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            iou_list.append(iou)

    return (
        total_loss,
        np.mean(precision_list),
        np.mean(recall_list),
        np.mean(f1_list),
        np.mean(iou_list)
    )



def compute_metrics(preds, targets):

    preds = torch.sigmoid(preds)

    preds = (preds > 0.5).float()
    
    preds = preds.squeeze(1)
    targets = targets.squeeze(1)

    preds = preds.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()

    preds = (preds > 0.5).astype(np.uint8)
    targets = (targets > 0.5).astype(np.uint8)

    precision = precision_score(
        targets,
        preds,
        zero_division=0
    )

    recall = recall_score(
        targets,
        preds,
        zero_division=0
    )

    f1 = f1_score(
        targets,
        preds,
        zero_division=0
    )

    intersection = np.logical_and(
        preds,
        targets
    ).sum()

    union = np.logical_or(
        preds,
        targets
    ).sum()

    iou = intersection / union if union > 0 else 0

    return precision, recall, f1, iou
