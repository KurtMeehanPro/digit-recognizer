"""Evaluate digit classifiers on held-out data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch


def build_epoch_metrics(epoch, train_loss_sum, val_loss_sum, train_correct,
                        val_correct, train_count, val_count):
    """Calculate average loss and accuracy for a completed training/validation epoch."""
    return {
        "epoch": epoch,
        "train_loss": train_loss_sum / train_count,
        "val_loss": val_loss_sum / val_count,
        "train_accuracy": train_correct / train_count,
        "val_accuracy": val_correct / val_count,
    }


def plot_validation_confusion_matrix(model, val_loader):
    """Return the validation confusion matrix and its figure for digits 0 through 9."""
    device = next(model.parameters()).device
    model.eval()
    actual_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            logits = model(batch_images.to(device))
            predicted_labels.extend(logits.argmax(dim=1).cpu().tolist())
            actual_labels.extend(batch_labels.cpu().tolist())

    matrix = confusion_matrix(actual_labels, predicted_labels, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(matrix, display_labels=list(range(10))).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False,
    )
    ax.set_title("Validation confusion matrix — best model")
    fig.tight_layout()
    return matrix, fig


def build_per_class_metrics(matrix):
    """Summarize a confusion matrix whose rows are actual classes and columns predictions.

    Count and incorrect predictions refer to actual examples of each class.
    Accuracy treats each class versus all others and includes true negatives.
    Undefined precision, recall, or F1 values are reported as zero.
    """
    matrix = np.asarray(matrix)
    count = matrix.sum(axis=1)
    correct = matrix.diagonal()
    predicted_count = matrix.sum(axis=0)
    total = matrix.sum()
    false_positive = predicted_count - correct
    false_negative = count - correct
    true_negative = total - correct - false_positive - false_negative

    def divide(numerator, denominator):
        return np.divide(
            numerator, denominator,
            out=np.zeros_like(numerator, dtype=float), where=denominator != 0,
        )

    return pd.DataFrame({
        "Class": np.arange(len(matrix)),
        "Count": count,
        "Correctly Predicted": correct,
        "Incorrectly Predicted": false_negative,
        "Precision": divide(correct, predicted_count),
        "Recall": divide(correct, count),
        "F1": divide(2 * correct, predicted_count + count),
        "Accuracy": divide(correct + true_negative, total),
    }).set_index("Class")


# image classification function
def plot_misclassified_digit(model, val_loader, digit=3, columns=8, max_images=None,
                             validation_logits=None):
    """Display errors for a digit, optionally capped in validation order.

    max_images=None preserves the original complete gallery. A cap limits only
    displayed images; the reported error count still covers the full loader.
    validation_logits can reuse aligned saved outputs without model inference;
    in that case model may be None and the loader must keep validation order.
    """
    if max_images is not None and (not isinstance(max_images, int) or max_images < 1):
        raise ValueError("max_images must be a positive integer or None")
    if validation_logits is None:
        device = next(model.parameters()).device
        model.eval()
    elif tuple(validation_logits.shape) != (len(val_loader.dataset), 10):
        raise ValueError("validation_logits must have shape (validation images, 10)")
    offset = 0
    images = []
    predictions = []
    total_errors = 0
    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            if validation_logits is None:
                predicted = model(batch_images.to(device)).argmax(dim=1).cpu()
            else:
                predicted = validation_logits[offset:offset + len(batch_labels)].argmax(1).cpu()
            offset += len(batch_labels)
            labels = batch_labels.cpu()
            mask = (labels == digit) & (predicted != labels)
            total_errors += int(mask.sum())
            remaining = int(mask.sum()) if max_images is None else max(0, max_images - len(images))
            images.extend(batch_images.cpu()[mask][:remaining])
            predictions.extend(predicted[mask][:remaining].tolist())

    print(f"Actual digit {digit}: {total_errors} misclassified validation images")
    if max_images is not None:
        print(f"Showing {len(images)} of {total_errors} errors (limit {max_images})")
    if not images:
        return None
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.6, rows * 1.9), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, prediction in zip(axes.flat, images, predictions):
        ax.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Predicted {prediction}")
    title = f"Misclassified validation images — actual digit {digit}"
    if max_images is not None:
        title += f": showing {len(images)} of {total_errors} errors"
    fig.suptitle(title)
    if max_images is None:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout(h_pad=2.0, rect=(0, 0, 1, 0.90))
    return fig


def aligned_saved_validation(artifact, frame):
    """Reconstruct upright validation rows; reject any prediction/image mismatch.

    source_row_index is the zero-based train.csv data row (header excluded);
    csv_line_number adds two. validation_index is the saved prediction position.
    """
    import hashlib
    from sklearn.model_selection import train_test_split
    from scripts.dataset.digit_dataset_config import PIXEL_COLUMNS

    IMAGE_RESHAPE = (-1, 1, 28, 28)
    digest = hashlib.sha256(frame[["label", *PIXEL_COLUMNS]].to_numpy().tobytes()).hexdigest()

    if digest != artifact["dataset_sha256"]:
        raise ValueError("Raw dataset fingerprint differs from saved run")

    _, indices = train_test_split(np.arange(len(frame)),
        test_size=artifact["validation_fraction"], random_state=artifact["seed"], stratify=frame["label"].to_numpy())

    images = torch.tensor(frame.iloc[indices][PIXEL_COLUMNS].to_numpy(), dtype=torch.float32).reshape(IMAGE_RESHAPE) / 255.0
    labels = torch.tensor(frame.iloc[indices]["label"].to_numpy(), dtype=torch.long)
    fingerprint = hashlib.sha256(images.numpy().tobytes() + labels.numpy().tobytes()).hexdigest()
    logits = artifact["logits"].detach().cpu()
    if not torch.equal(labels, artifact["labels"].cpu()) or fingerprint != artifact["validation_sha256"]:
        raise ValueError("Saved labels/order do not match reconstructed upright validation images")
    if tuple(logits.shape) != (len(labels), 10) or not torch.isfinite(logits).all():
        raise ValueError("Invalid saved validation logits")
    return images, labels, indices


def build_error_tables(logits, labels, source_indices):
    """Saved-logit analysis only. Softmax probability is not calibrated certainty."""
    probabilities = logits.softmax(1)
    confidence, predicted = probabilities.max(1)
    correct = predicted.eq(labels)
    matrix = confusion_matrix(labels.numpy(), predicted.numpy(), labels=list(range(10)))
    per_digit = build_per_class_metrics(matrix).rename(columns={"Count":"Support", "Incorrectly Predicted":"Errors"})
    per_digit = per_digit[["Support", "Correctly Predicted", "Errors", "Precision", "Recall", "F1"]]
    top_two = probabilities.topk(2,dim=1).values
    predictions = pd.DataFrame({"validation_index":np.arange(len(labels)), "source_row_index":source_indices,
        "csv_line_number":np.asarray(source_indices)+2, "true_digit":labels.numpy(), "predicted_digit":predicted.numpy(),
        "confidence":confidence.numpy(), "true_class_probability":probabilities[torch.arange(len(labels)),labels].numpy(),
        "top_two_margin":(top_two[:,0]-top_two[:,1]).numpy(), "correct":correct.numpy()})
    errors = predictions.loc[~predictions.correct].copy()
    pairs = pd.DataFrame([{"true_digit":i,"predicted_digit":j,"errors":int(matrix[i,j])}
                          for i in range(10) for j in range(10) if i!=j and matrix[i,j]],
                         columns=["true_digit","predicted_digit","errors"])
    pairs = pairs.sort_values(["errors","true_digit","predicted_digit"],ascending=[False,True,True])
    bins=[]
    identifiers=np.searchsorted([.5,.7,.9,.99],predictions.confidence.to_numpy(),side="right")
    for index,label in enumerate(["<50%","50–<70%","70–<90%","90–<99%","99–100%"]):
        subset=predictions.loc[identifiers==index]
        bins.append({"predicted_confidence":label,"count":len(subset),"errors":int((~subset.correct).sum()),
                     "accuracy":float(subset.correct.mean()) if len(subset) else None,
                     "mean_confidence":float(subset.confidence.mean()) if len(subset) else None})
    return matrix,per_digit,predictions,errors,pairs,pd.DataFrame(bins)


def plot_error_gallery(images, examples, title, columns=4):
    """Compact deterministic galleries; examples contain unique validation indices."""
    if examples.validation_index.duplicated().any():
        raise ValueError("Duplicate examples in gallery")
    rows=max(1,(len(examples)+columns-1)//columns)
    fig,axes=plt.subplots(rows,columns,figsize=(columns*1.85,rows*2.5+.7),squeeze=False)
    for ax in axes.flat:ax.axis("off")
    if examples.empty:
        axes[0,0].text(0,.5,"No misclassified examples",fontsize=11)
    for ax,(_,row) in zip(axes.flat,examples.iterrows()):
        ax.imshow(images[int(row.validation_index),0].numpy(),cmap="gray",vmin=0,vmax=1,interpolation="nearest")
        ax.set_title(f"True {int(row.true_digit)} → {int(row.predicted_digit)}\np={row.confidence:.1%}\nrow {int(row.source_row_index)}",fontsize=9)
    fig.suptitle(title,fontsize=11)
    fig.subplots_adjust(left=.02,right=.98,bottom=.04,top=.80,hspace=.8,wspace=.12)
    return fig


def investigate_saved_errors(validation_artifact, output_directory=None, max_per_digit=8, display_report=True):
    """Generate error artifacts without model inference or training.

    First errors in validation order are shown per true digit, never selected for
    apparent difficulty. A separate confidence-sorted table covers every error.
    Run/checkpoint, dataset and exact image/label/order fingerprints are recorded.
    Existing output directories are read-only reports and never overwritten.
    """
    import hashlib,json
    from pathlib import Path
    from datetime import datetime,timezone
    from uuid import uuid4
    from scripts.dataset.digit_dataset_config import PROJECT_ROOT
    if not isinstance(max_per_digit,int) or not 1<=max_per_digit<=8:
        raise ValueError("max_per_digit must be an integer from 1 through 8")
    path=Path(validation_artifact).resolve()
    from utilities.digit_run_artifacts import load_run, validation_path
    path=validation_path(path).resolve()
    artifact=load_run(path)
    if not Path(artifact["checkpoint_path"]).is_file():
        raise ValueError("Referenced saved checkpoint is missing")
    frame=pd.read_csv(PROJECT_ROOT/'data/raw/train.csv')
    images,labels,source_indices=aligned_saved_validation(artifact,frame)
    matrix,per_digit,predictions,errors,pairs,bins=build_error_tables(artifact['logits'],labels,source_indices)
    if output_directory is None:
        directory = path.parent if path.name == 'output.pt' else Path(artifact['sweep_directory'])/'error_investigations'/('saved_validation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    else:directory=Path(output_directory)
    if (directory/'summary.json').is_file():
        metadata=json.loads((directory/'summary.json').read_text())
        if metadata['validation_artifact_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest() or metadata['max_per_digit']!=max_per_digit:
            raise ValueError("Existing report belongs to different predictions/settings")
    else:
        directory.mkdir(parents=True,exist_ok=True)
        per_digit.to_csv(directory/'per_digit_metrics.csv')
        predictions.to_csv(directory/'validation_predictions.csv',index=False)
        pairs.to_csv(directory/'confusion_pairs.csv',index=False)
        errors.sort_values(['confidence','validation_index'],ascending=[False,True]).to_csv(directory/'confident_errors.csv',index=False)
        bins.to_csv(directory/'confidence_summary.csv',index=False)
        np.savetxt(directory/'confusion_matrix.csv',matrix,fmt='%d',delimiter=',')
        fig,ax=plt.subplots(figsize=(8,7.5))
        ConfusionMatrixDisplay(matrix,display_labels=list(range(10))).plot(ax=ax,cmap='Blues',values_format='d',colorbar=False)
        ax.set_title(f"Saved {artifact['model_config']['activation'].upper()} validation — {int(len(labels)-len(errors)):,}/{len(labels):,} correct")
        fig.tight_layout();fig.savefig(directory/'confusion_matrix.png',dpi=160);plt.close(fig)
        shown=[]
        for digit in range(10):
            digit_errors=errors.loc[errors.true_digit==digit]
            examples=digit_errors.head(max_per_digit)
            shown.extend(examples.validation_index.tolist())
            fig=plot_error_gallery(images,examples,f"True digit {digit}: showing {len(examples)} of {len(digit_errors)} misses")
            fig.savefig(directory/f'true_digit_{digit}_misses.png',dpi=160);plt.close(fig)
        # Overview contains the same examples as the individual digit galleries.
        fig,axes=plt.subplots(10,max_per_digit,figsize=(max_per_digit*1.65,17.5),squeeze=False)
        for digit in range(10):
            examples=errors.loc[errors.true_digit==digit].head(max_per_digit)
            total=int((errors.true_digit==digit).sum())
            for ax in axes[digit]:ax.axis('off')
            for ax,(_,row) in zip(axes[digit],examples.iterrows()):
                ax.imshow(images[int(row.validation_index),0].numpy(),cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                ax.set_title(f"{digit} → {int(row.predicted_digit)} · {row.confidence:.1%}\nrow {int(row.source_row_index)}",fontsize=8)
            axes[digit,0].text(-.18,.5,f"True {digit}\n{len(examples)}/{total}",transform=axes[digit,0].transAxes,ha='right',va='center',fontsize=9)
        fig.suptitle('Saved winner: first up to eight misses per true digit\nModel probabilities are not calibrated certainty; blank slots mean fewer available errors.',fontsize=12)
        fig.tight_layout(rect=(.035,0,1,.96));fig.savefig(directory/'all_digit_misses.png',dpi=180);plt.close(fig)
        summary={"run_id":artifact['run_id'],"checkpoint_path":artifact['checkpoint_path'],"selected_epoch":artifact['best_epoch'],
            "selection":"Checkpoint selected by minimum validation loss within saved run",
            "model_config":artifact["model_config"],"selected_validation_loss":artifact["history"][artifact["best_epoch"]-1]["val_loss"],
            "validation_artifact":str(path),"validation_artifact_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "checkpoint_sha256":hashlib.sha256(Path(artifact['checkpoint_path']).read_bytes()).hexdigest(),
            "dataset_sha256":artifact['dataset_sha256'],"validation_sha256":artifact['validation_sha256'],
            "validation_count":len(labels),"errors":len(errors),"correct":len(labels)-len(errors),"accuracy":float(predictions.correct.mean()),
            "errors_by_true_digit":{str(d):int((errors.true_digit==d).sum()) for d in range(10)},
            "max_per_digit":max_per_digit,"displayed_unique_errors":len(shown),"displayed_validation_indices":shown,
            "gallery_selection":"first errors in validation order per true digit; overview repeats individual-gallery examples",
            "source_index_convention":"zero-based train.csv data row; csv_line_number=source_row_index+2; validation_index is saved prediction order",
            "confidence_note":"softmax probability, not calibrated certainty",
            "wrong_at_least_90_percent":int((errors.confidence>=.9).sum()),"wrong_at_least_99_percent":int((errors.confidence>=.99).sum()),
            "mean_error_confidence":float(errors.confidence.mean()) if len(errors) else None,
            "median_error_confidence":float(errors.confidence.median()) if len(errors) else None,
            "macro_f1":float(per_digit.F1.mean()),"training_or_inference_executed":False}
        (directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if display_report:
        from IPython.display import display,Image,Markdown
        display(Markdown(f"### Error investigation — {artifact['run_id']} · selected epoch {artifact['best_epoch']}"))
        print('Saved checkpoint:',artifact['checkpoint_path'])
        print(f'Selection: saved run, minimum validation loss; accuracy {predictions.correct.mean():.2%}.')
        print('Confidence is model softmax probability, not calibrated certainty. Row IDs are zero-based source CSV data rows.')
        display(Image(filename=str(directory/'confusion_matrix.png'),width=650));display(per_digit)
        display(pairs.head(12));display(bins)
        display(errors.sort_values(['confidence','validation_index'],ascending=[False,True]).head(12))
        for digit in range(10):display(Image(filename=str(directory/f'true_digit_{digit}_misses.png'),width=740))
    return directory
