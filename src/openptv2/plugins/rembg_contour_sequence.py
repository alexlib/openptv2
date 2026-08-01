import os
from pathlib import Path

import numpy as np
from imageio.v3 import imread

from openptv2.correspondences import MatchedCoords, correspondences
from openptv2.orientation import point_positions
from openptv2.tracker import default_naming

# rembg (and its ONNX model download) is only imported on first actual use —
# not a core dependency, install with `openptv2[rembg]`.
_session = None


def _get_session():
    global _session
    if _session is None:
        from rembg import new_session

        _session = new_session("u2net")
    return _session


def save_mask_areas(areas_data: list, output_file: Path) -> None:
    """Save mask areas to CSV file.

    Parameters
    ----------
    areas_data : list
        List of dictionaries containing camera number, frame number, and area
    output_file : Path
        Path to output CSV file
    """
    import pandas as pd

    df = pd.DataFrame(areas_data)
    df.to_csv(output_file, index=False)


def mask_image(imname: Path, display: bool = False) -> tuple[np.ndarray, float]:
    """Mask the image using rembg and keep the entire mask.

    Parameters
    ----------
    imname : Path
        Path to the image file
    display : bool
        Whether to display debug plots

    Returns
    -------
    tuple[np.ndarray, float]
        Masked image and the area of the mask below row 600 in pixels
    """
    from rembg import remove

    input_data = imread(imname)
    mask = remove(input_data, session=_get_session(), only_mask=True)

    # Set ROI threshold
    y_threshold = 600

    # Create ROI mask below threshold
    roi_mask = np.zeros_like(mask, dtype=bool)
    roi_mask[y_threshold:, :] = True

    # Calculate area in ROI
    mask_in_roi = np.where(roi_mask, mask, False)
    area = np.sum(mask_in_roi)

    if display:
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        ax1.imshow(input_data)
        ax1.axhline(y=y_threshold, color="r", linestyle="--")
        ax1.set_title("Original image")

        ax2.imshow(mask)
        ax2.axhline(y=y_threshold, color="r", linestyle="--")
        ax2.set_title("Full mask")

        ax3.imshow(np.where(mask, input_data, 0))
        ax3.axhline(y=y_threshold, color="r", linestyle="--")
        ax3.set_title("Masked image")

        ax4.imshow(np.where(mask_in_roi, input_data, 0))
        ax4.set_title(f"ROI mask (area: {area} pixels)")

        plt.tight_layout()
        plt.show()

    # Apply the mask to the input image
    masked_image = np.where(mask, input_data, 0)
    return masked_image, area


class Sequence:
    """Sequence plugin that removes the background with ``rembg``, tracks
    the mask area per frame, and writes it to ``res/mask_areas.csv``.

    Connection to the ptv module is given via ``self.ptv`` and connection to
    the active experiment via ``self.exp``, both injected by the loader.
    """

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp
        self.areas_data = []  # Store areas data during processing

    def do_sequence(self):
        num_cams, cpar, spar, vpar, tpar, cals = (
            self.exp.num_cams,
            self.exp.cpar,
            self.exp.spar,
            self.exp.vpar,
            self.exp.tpar,
            self.exp.cals,
        )

        first_frame = spar.get_first()
        last_frame = spar.get_last()
        print(f" From {first_frame = } to {last_frame = }")

        for frame in range(first_frame, last_frame + 1):
            detections = []
            corrected = []
            for i_cam in range(num_cams):
                base_image_name = spar.get_img_base_name(i_cam)
                imname = Path(base_image_name % frame)  # works with jumps from 1 to 10
                masked_image, area = mask_image(imname, display=False)

                self.areas_data.append({"camera": i_cam, "frame": frame, "area": area})

                high_pass = self.ptv.simple_highpass(masked_image, cpar)
                targs = self.ptv.target_recognition(high_pass, tpar, i_cam, cpar)

                targs.sort_y()
                detections.append(targs)
                masked_coords = MatchedCoords(targs, cpar, cals[i_cam])
                pos, _ = masked_coords.as_arrays()
                corrected.append(masked_coords)

            # Corresp. + positions.
            sorted_pos, sorted_corresp, _ = correspondences(
                detections, corrected, cals, vpar, cpar
            )

            # Save targets only after they've been modified:
            for i_cam in range(num_cams):
                base_name = spar.get_img_base_name(i_cam)
                self.ptv.write_targets(detections[i_cam], base_name, frame)

            print(
                "Frame "
                + str(frame)
                + " had "
                + repr([s.shape[1] for s in sorted_pos])
                + " correspondences."
            )

            # Distinction between quad/trip irrelevant here.
            sorted_pos = np.concatenate(sorted_pos, axis=1)
            sorted_corresp = np.concatenate(sorted_corresp, axis=1)

            flat = np.array(
                [corrected[i].get_by_pnrs(sorted_corresp[i]) for i in range(len(cals))]
            )
            pos, _ = point_positions(flat.transpose(1, 0, 2), cpar, cals, vpar)

            if len(cals) < 4:
                print_corresp = -1 * np.ones((4, sorted_corresp.shape[1]))
                print_corresp[: len(cals), :] = sorted_corresp
            else:
                print_corresp = sorted_corresp

            storage_mode = os.environ.get("OPENPTV_STORAGE", "zarr").lower()
            if storage_mode in ("zarr", "zarr_only"):
                from openptv2.storage import ZarrFrameStore

                zarr_path = Path("res/run.zarr")
                zarr_path.parent.mkdir(parents=True, exist_ok=True)
                store = ZarrFrameStore(zarr_path, mode="a")
                store.write_correspondences(
                    frame=frame, pos_3d=pos, cam_target_ids=print_corresp.T
                )

            if storage_mode != "zarr_only":
                # Save rt_is
                rt_is_filename = default_naming["corres"]
                if isinstance(rt_is_filename, bytes):
                    rt_is_filename = rt_is_filename.decode("utf-8")
                rt_is_filename = f"{rt_is_filename}.{frame}"
                with open(rt_is_filename, "w", encoding="utf8") as rt_is:
                    rt_is.write(str(pos.shape[0]) + "\n")
                    for pix, pt in enumerate(pos):
                        pt_args = (pix + 1,) + tuple(pt) + tuple(print_corresp[:, pix])
                        rt_is.write("%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n" % pt_args)

        # After processing all frames, save the areas data
        output_file = Path("res/mask_areas.csv")
        save_mask_areas(self.areas_data, output_file)
        print(f"Mask areas saved to {output_file}")
