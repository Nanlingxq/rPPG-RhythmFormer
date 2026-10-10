# VIPL-HR low-heart-rate augmentation

`TRAIN.LOW_HR_AUG_MODE` selects the low-HR augmentation used by the
RhythmFormer trainer for VIPL-HR:

```yaml
TRAIN:
  LOW_HR_AUG_MODE: legacy_repeat  # original decimate-and-repeat behavior
```

Set the value to `continuous_320` to form a 320-frame sequence from the
current clip and its next clip from the same recording, then sample every
other frame to obtain 160 frames. If a same-recording next clip is not
available, the trainer falls back to `legacy_repeat` and reports the fallback
count. The mode defaults to `legacy_repeat`, preserving existing runs.

The two supported values are `legacy_repeat` and `continuous_320`.
