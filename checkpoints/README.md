# Checkpoints

| File | What | How it was produced |
|---|---|---|
| `classifiers.pt` / `.json` | Genre + emotion MLP over 70 key-invariant features | `python -m midicomposer.cli train-classifiers --data data/xmidi` on the full train split (77,086 files); metrics on held-out val/test in the JSON |
| `composer_60_40.pt` / `.json` | HierarchicalComposer, 6.8M params. **The model the app uses.** | `python scripts/resplit_manifest.py --out data/xmidi_60_40 --train 0.57 --val 0.03` then `python -m midicomposer.cli train --config configs/train_60_40.yaml`: all 54,824 train files of the 60/40 split, 40,000 steps (best at 39,000), best val loss 0.600. Test loss 0.590 on the 40% test split (`python scripts/eval_test_loss.py`, `reports/test_eval_60_40.json`). |
| `composer_dev.pt` / `.json` | HierarchicalComposer, 6.8M params (same architecture). Kept for comparison. | `python -m midicomposer.cli train --config configs/dev_train.yaml`: 8,000 of 77,086 usable train files, 4,000 steps, best val loss 1.234. **Development-scale, not fully trained.** Test loss 1.233 on the same held-out files as above. |

Every model checkpoint's JSON contains `train_files`, `usable_train_files_in_manifest`,
`trained_on_all_usable_train_files` and an `honesty_note`; the API and generation manifests surface that note.
No weights in this folder were obtained any other way. The one edit after training: `composer_60_40`'s
`honesty_note` was rewritten with the current `train.honesty_note()` wording (it now says the training files are 57%
of usable XMIDI, not all of it); its weights are bit-identical to the trained ones.
