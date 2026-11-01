# Saved conventional ReLU winner — error investigation

**Status / selection:** Complete saved-output analysis, October 30, 2026. Run `20261030_143937_9823b3b6`, LR0.00075, three epochs, selected epoch3 by minimum validation loss. Chosen saved 99.14% augmented conventional ReLU winner; no training or inference performed.

**Alignment:** Reconstructed the original seed42 stratified 80/20 split before augmentation. Raw dataset hash, upright validation image/label hash and label order match saved predictions. Stable IDs include zero-based source CSV data row, CSV line (+2), and saved validation position.

**Results:** 8,328/8,400 correct; 72 misses; macro F1 99.13%. Errors for true digits 0–9: **2, 2, 11, 9, 6, 17, 4, 9, 4, 8**. Galleries show the first up to eight per true digit in validation order: 58 unique misses; no duplicated examples within/between digit groups. Overview repeats those same galleries. All 72 errors remain in CSV.

**Patterns:** Digit5 has the most errors (17/759; recall97.76%): 5→6 six times, 5→3 and 5→8 five each. Other common pairs include 2→7 (5), 4→9 (4), 7→1 (4). Wrong predictions include 23 at ≥90% probability, seven at ≥99%; median error probability75.61%. Confidence is softmax model probability, not calibrated certainty.

**Visual review:** The shown errors mix unusual/ambiguous writing with recognizable digits; this does not establish that all remaining errors are intrinsically hard. Source row3436 looks recognizable as5 but was predicted3 at99.4%; row24477 is labeled3 but resembles5 to this reviewer. These are subjective observations, not corrected labels or a blinded human recognition study. No labels were changed.

**Artifacts / use:** `data/experiments/bespoke_lr_sweep_20261030_143704_01fa13bf/error_investigations/saved_validation_20261030T203803Z_b155aef1/` contains confusion matrix PNG/CSV, per-digit support/precision/recall/F1/errors, ten compact digit galleries, an overview, all aligned predictions, confidence-sorted errors, confusion-pair counts, confidence bins, provenance and verification. Notebook05 ends with a short saved-winner reporting call, independently runnable after setup; all six existing experiment sections preserved. No prior results overwritten.

**Delivery:** Local images verified. Native Library upload failed before preparation because of DNS; no Library files/IDs were created. Local PNGs remain available. No commit or push.
