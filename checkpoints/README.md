# Checkpoints

| File | What | How it was produced |
|---|---|---|
| `classifiers.pt` / `.json` | Genre + emotion MLP over 70 key-invariant features | `python -m midicomposer.cli train-classifiers --data data/xmidi` on the full train split (77,086 files); metrics on held-out val/test in the JSON |
| `composer_dev.pt` / `.json` | HierarchicalComposer, 6.8M params | `python -m midicomposer.cli train --config configs/dev_train.yaml`: 8,000 of 77,086 usable train files, 4,000 steps, best val loss 1.234. **Development-scale, not fully trained.** |

Every model checkpoint's JSON contains `train_files`, `usable_train_files_in_manifest`,
`trained_on_all_usable_train_files` and an `honesty_note`; the API and generation manifests surface that note.
No weights in this folder were obtained any other way.
