# Sample batch files

Upload these from the dashboard (Details → Simulate data → Upload a batch file),
or edit them in Excel and upload your own.

| File | What it simulates | Expected result |
|---|---|---|
| `<disease>_clean.csv` | new patients like the training population | within limits |
| `<disease>_drifted.csv` | key measurements shifted by 1.5 standard deviations | data drift detected |
| `<disease>_concept_shift.csv` | the measurement–outcome relationship changed | performance drop, then retraining |
| `<disease>_unlabelled.csv` | new patients without outcomes (no `Outcome` column) | drift check only |

Rules for your own files:

* One header row with every feature column of that disease (extra columns are ignored).
* `Outcome` is optional: `1` = disease present, `0` = not present. Without it the
  batch is checked for drift only and is not used for retraining.
* At least 30 patient rows. Empty cells are allowed (they are filled in the same
  way as during training). Comma- or semicolon-separated both work.

The rows come from each disease's held-out "future pool": patients the models
never saw during training, validation or testing.
