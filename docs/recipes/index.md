# Recipes: what to do in your situation

Each recipe starts from what you have and what you want, and sends you to the [course](../course/index.md) for the basic
steps instead of repeating them. Recipes are named for the situation, not for the lab where the data came from.

| You have ... | You want ... | Recipe |
|---|---|---|
| two rigs of four cameras each and a hand-held dots plate | a calibration for each rig (OpenCV-style free-pose plate calibration), then tracking | [Two rigs of four cameras, hand-held plate](plate-calibration-two-rigs.md) |
| one rig and a target with known 3D points | the classic calibration | [One rig, a target with known 3D points](calibration-known-target.md) |
| cameras on both sides of a water cell | to track through two windows | [Cameras on both sides of a water cell](water-cell-two-sides.md) |
| a calibration you are not sure about | to know whether it is good enough | [Course page 8: is the calibration good?](../course/08-calibration.md) |
| tracks that break when the camera set changes | to find out if the cameras are the cause, and repair it | [Tracks break when the set of cameras changes](camera-set-changes.md) |
| dense seeding, merged blobs | more tracks | [Dense seeding](dense-seeding.md) |
| many runs sharing one calibration, or a server | to process in batch | [Many runs and servers](many-runs-and-servers.md) |
| a dumbbell | to calibrate or check scale | [Dumbbell](dumbbell.md) |
