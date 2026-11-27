# Recipes: what to do in your situation

Each recipe starts from what you have and what you want, and sends you to the [course](../course/index.md) for the basic
steps instead of repeating them. Recipes are named for the situation, not for the lab where the data came from.

| You have ... | You want ... | Recipe |
|---|---|---|
| two rigs of four cameras each and a hand-held dots plate | a calibration for each rig (OpenCV-style free-pose plate calibration), then tracking | [Two rigs of four cameras, hand-held plate](plate-calibration-two-rigs.md) |
| one rig and a plate moved by known steps | the classic multi-plane calibration | [Camera calibration](../tutorials/calibration.md) |
| cameras on both sides of a water cell | to track through two windows | *in preparation*; see [Synthetic cavity study](../self-calibration-cavity-synthetic.md) |
| a calibration you are not sure about | to know whether it is good enough | *in preparation*; see [Calibration consistency check](../calibration-consistency.md) |
| tracks that break when the camera set changes | to repair that | *in preparation*; see [Calibration consistency check](../calibration-consistency.md) |
| dense seeding, merged blobs | more tracks | *in preparation*; see [Holistic tracking](../holistic-tracking.md) |
| many runs sharing one calibration, or a server | to process in batch | [Multi-folder runs](../multi-folder-runs.md), [Cloud-like batch](../cloud-batch.md) |
| a dumbbell | to calibrate or check scale | [Dumbbell calibration](../tutorials/dumbbell_calibration.md) |
