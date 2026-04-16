# Project Status - C to Python Translation

## Summary
Translating the OpenPTV C library (`lib/src/**`) into pure Python with NumPy (`algorithms/**`) following a direct, SoA-based approach.

## 📝 Translation Progress

| Module | Status | Notes |
| :--- | :--- | :--- |
| `vec_utils` | ✅ Complete | Fully translated and tested. |
| `lsqadj` | ✅ Complete | Fully translated and tested. |
| `calibration` | ✅ Complete | Fully translated and tested. |
| `parameters` | ✅ Complete | Fully translated and tested. |
| `trafo` | ✅ Complete | Fully translated and tested. |
| `multimed` | ✅ Complete | Fully translated and tested. |
| `ray_tracing` | ✅ Complete | Fully translated and tested. |
| `imgcoord` | ✅ Complete | Fully translated and tested. |
| `image_processing` | ✅ Complete | Fully translated and tested. |
| `epi` | ✅ Complete | Fully translated and tested. |
| `orientation` | ✅ Complete | Fully translated and tested. |
| `correspondences` | ✅ Complete | Fully translated and tested. |
| `segmentation` | ⏳ In Progress | `targ_rec` done, `peak_fit` needs `check_touch`. |
| `sortgrid` | 📝 Pending | |
| `tracking_frame_buf`| 📝 Pending | |
| `tracking_run` | 📝 Pending | |
| `track` | 📝 Pending | |
| `track3d` | 📝 Pending | |

## 🚀 Next Step
Implement `check_touch` in `algorithms/segmentation.py` and verify `segmentation` tests.
