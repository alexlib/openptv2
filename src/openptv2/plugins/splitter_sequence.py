from pathlib import Path

import numpy as np
from imageio.v3 import imread

from openptv2.correspondences import MatchedCoords, correspondences
from openptv2.orientation import point_positions
from openptv2.tracker import default_naming


class Sequence:
    """Sequence plugin for four-view-splitter cameras: reads a single frame
    per multiplexed camera and splits it into per-view images with
    ``ptv.image_split`` before detection and correspondence.

    Connection to the ptv module is given via ``self.ptv`` and connection to
    the active experiment via ``self.exp``, both injected by the loader.
    """

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_sequence(self):
        if self.exp is None:
            raise ValueError("No experiment object provided")

        if hasattr(self.exp, "ensure_parameter_objects"):
            self.exp.ensure_parameter_objects()

        # Verify splitter mode is enabled
        if hasattr(self.exp, "pm"):
            ptv_params = self.exp.pm.get_parameter("ptv")
            if not ptv_params.get("splitter", False):
                raise ValueError(
                    "Splitter mode must be enabled for this sequence processor"
                )

            masking_params = self.exp.pm.get_parameter("masking")
            inverse_flag = ptv_params.get("inverse", False)
        else:
            # Fallback for older experiment objects
            masking_params = {}
            inverse_flag = False

        required_attrs = ["cpar", "spar", "vpar", "tpar", "cals"]
        if not all(hasattr(self.exp, attr) for attr in required_attrs):
            raise ValueError("Experiment object missing required parameter objects")

        num_cams = len(self.exp.cals)
        cpar = self.exp.cpar
        spar = self.exp.spar
        vpar = self.exp.vpar
        tpar = self.exp.tpar
        cals = self.exp.cals

        first_frame = spar.get_first()
        last_frame = spar.get_last()
        print(f" From {first_frame = } to {last_frame = }")

        for frame in range(first_frame, last_frame + 1):
            print(f"Processing frame {frame}")

            detections = []
            corrected = []

            # when we work with splitter, we read only one image
            base_image_name = spar.get_img_base_name(0)

            if isinstance(base_image_name, bytes):
                base_image_name = base_image_name.decode("utf-8")

            print(
                f"Base image name: '{base_image_name}' "
                f"(type: {type(base_image_name)}) for frame {frame}"
            )

            try:
                imname = Path(base_image_name % frame)  # works with jumps from 1 to 10
                print(f"Formatted image name: {imname}")
            except (TypeError, ValueError) as e:
                print(
                    f"String formatting failed for '{base_image_name}' "
                    f"with frame {frame}: {e}"
                )
                # Fallback: assume base_image_name is already formatted or
                # needs frame appended
                if "%" not in base_image_name:
                    base_path = Path(base_image_name)
                    imname = (
                        base_path.parent
                        / f"{base_path.stem}_{frame:04d}{base_path.suffix}"
                    )
                    print(f"Using fallback image name: {imname}")
                else:
                    raise ValueError(
                        "String formatting error with base_image_name "
                        f"'{base_image_name}': {e}"
                    )

            if not imname.exists():
                raise FileNotFoundError(f"{imname} does not exist")

            # now we read and split
            full_image = imread(imname)
            if full_image.ndim > 2:
                from skimage.color import rgb2gray

                full_image = rgb2gray(full_image)

            if inverse_flag:
                full_image = self.ptv.negative(full_image)

            # Split image using configurable order (HI-D specific order)
            list_of_images = self.ptv.image_split(full_image, order=[0, 1, 3, 2])

            for i_cam in range(num_cams):
                masked_image = list_of_images[i_cam].copy()

                if masking_params.get("mask_flag", False):
                    try:
                        mask_base_name = masking_params.get("mask_base_name", "")
                        if not mask_base_name:
                            print(
                                "Warning: mask_flag is True but "
                                "mask_base_name is empty"
                            )
                            continue

                        if "%" in mask_base_name:
                            background_name = mask_base_name % (i_cam + 1)
                        else:
                            mask_path = Path(mask_base_name)
                            background_name = str(
                                mask_path.parent
                                / f"{mask_path.stem}_cam{i_cam + 1}{mask_path.suffix}"
                            )

                        background = imread(background_name)
                        if background.ndim > 2:
                            from skimage.color import rgb2gray

                            background = rgb2gray(background)
                        masked_image = np.clip(
                            masked_image - background, 0, 255
                        ).astype(np.uint8)
                    except (ValueError, FileNotFoundError, TypeError) as e:
                        print(f"Failed to read/apply mask for camera {i_cam}: {e}")

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

            # Save targets only after they've been modified (short file base names):
            for i_cam in range(num_cams):
                base_name = self.exp.target_filenames[i_cam]
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

            # Handle fewer than 4 cameras case
            if len(cals) < 4:
                print_corresp = -1 * np.ones((4, sorted_corresp.shape[1]))
                print_corresp[: len(cals), :] = sorted_corresp
            else:
                print_corresp = sorted_corresp

            # Save rt_is
            rt_is_filename = default_naming["corres"]
            if isinstance(rt_is_filename, bytes):
                rt_is_filename = rt_is_filename.decode("utf-8")
            rt_is_filename = f"{rt_is_filename}.{frame}"
            with open(rt_is_filename, "w", encoding="utf8") as rt_is:
                rt_is.write(str(pos.shape[0]) + "\n")
                for pix, pt in enumerate(pos):
                    try:
                        pt_args = (pix + 1,) + tuple(pt) + tuple(print_corresp[:, pix])
                        if len(pt_args) != 8:
                            print(
                                f"Warning: pt_args has {len(pt_args)} elements, "
                                "expected 8"
                            )
                            print(f"pt_args = {pt_args}")
                        rt_is.write("%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n" % pt_args)
                    except (TypeError, ValueError) as e:
                        print(
                            f"String formatting error at frame {frame}, "
                            f"pixel {pix}: {e}"
                        )
                        print(
                            f"pt = {pt}, "
                            f"print_corresp[:, {pix}] = {print_corresp[:, pix]}"
                        )
                        raise

        print("Sequence completed successfully")
