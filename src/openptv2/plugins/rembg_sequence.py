from pathlib import Path

import numpy as np
from imageio.v3 import imread
from skimage import img_as_ubyte
from skimage.color import rgb2gray

from openptv2.correspondences import MatchedCoords, correspondences
from openptv2.orientation import point_positions

# rembg (and its ONNX model download) is only imported on first actual use —
# not a core dependency, install with `openptv2[rembg]`.
_session = None


def _get_session():
    global _session
    if _session is None:
        from rembg import new_session

        _session = new_session("u2net")
    return _session


def mask_image(imname: Path, display: bool = False) -> np.ndarray:
    """Mask the image using a simple high pass filter.

    Parameters
    ----------
    img : np.ndarray
        The image to be masked.

    Returns
    -------
    np.ndarray
        The masked image.
    """
    from rembg import remove

    input_data = imread(imname)
    result = remove(input_data, session=_get_session())
    result = img_as_ubyte(rgb2gray(result[:, :, :3]))

    return result


class Sequence:
    """Sequence plugin that removes the background with ``rembg`` before
    detection and correspondence.

    Connection to the ptv module is given via ``self.ptv`` and connection to
    the active experiment via ``self.exp``, both injected by the loader.
    """

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

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

        store = self.ptv._open_run_store(self.exp)

        for frame in range(first_frame, last_frame + 1):
            detections = []
            corrected = []
            for i_cam in range(num_cams):
                base_image_name = spar.get_img_base_name(i_cam)
                imname = Path(base_image_name % frame)  # works with jumps from 1 to 10
                masked_image = mask_image(imname)

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
                base_name = self.exp.target_filenames[i_cam]
                self.ptv.write_targets(
                    detections[i_cam], base_name, frame, store=store, cam_idx=i_cam
                )

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

            store.write_correspondences(
                frame=frame, pos_3d=pos, cam_target_ids=print_corresp.T
            )
