from pathlib import Path

import numpy as np
from imageio.v3 import imread, imwrite
from skimage import filters, img_as_ubyte, measure, morphology
from skimage.color import label2rgb, rgb2gray
from skimage.morphology import binary_dilation, binary_erosion, disk

from openptv2.correspondences import MatchedCoords, correspondences
from openptv2.orientation import point_positions


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

    img = imread(imname)
    if img.ndim > 2:
        img = rgb2gray(img)

    if img.dtype != np.uint8:
        img = img_as_ubyte(img)

    # Apply Gaussian filter to smooth the image
    smoothed_frame = filters.gaussian(img, sigma=5)

    if display:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(smoothed_frame)
        plt.show()

    # Apply Otsu's thresholding method to segment the object
    thresh = filters.threshold_otsu(smoothed_frame)
    binary_frame = smoothed_frame > 1.1 * thresh

    if display:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(binary_frame)
        plt.show()

    binary_frame_cleared = binary_frame.copy()

    # Remove small bright objects
    cleaned_frame = morphology.remove_small_objects(
        binary_frame_cleared, min_size=100000
    )

    # Apply morphological closing to close the boundary
    closed_cleaned_frame = binary_dilation(cleaned_frame, disk(21))
    closed_cleaned_frame = binary_erosion(closed_cleaned_frame, disk(21))

    if display:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(closed_cleaned_frame, cmap="gray")
        plt.title("Closed Boundary of Cleaned Frame")
        plt.show()

    # Fill holes inside the binary frame to remove large black objects
    filled_frame = morphology.remove_small_holes(
        closed_cleaned_frame, area_threshold=2e6
    )

    if display:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(filled_frame, cmap="gray")
        plt.title("Binary Frame with Large Black Objects Removed")
        plt.show()

    # Label the segmented regions
    labeled_frame = measure.label(filled_frame)

    if display:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(label2rgb(labeled_frame, image=img, bg_label=0))
        plt.title("Color Labeled Frame with Filled Holes")
        plt.show()

    # Find region properties
    regions = measure.regionprops(labeled_frame)

    # Assuming the largest region is the object of interest
    largest_region = max(regions, key=lambda r: r.area)

    # Find the smooth contour that surrounds the largest region
    smooth_contour = morphology.convex_hull_image(largest_region.image)

    # Create an empty image to draw the smooth contour
    smooth_contour_image = np.zeros_like(labeled_frame, dtype=bool)

    # Place the smooth contour in the correct location
    minr, minc, maxr, maxc = largest_region.bbox
    smooth_contour_image[minr:maxr, minc:maxc] = smooth_contour

    if display:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(labeled_frame, cmap="jet")
        plt.contour(smooth_contour_image, colors="red", linewidths=2)
        plt.title("Segmented Object with Smooth Contour")
        plt.show()

    # Convert the largest region to a black and white image
    bw_image = np.zeros_like(labeled_frame, dtype=bool)
    bw_image[largest_region.coords[:, 0], largest_region.coords[:, 1]] = True

    # Apply morphological closing to remove sharp spikes
    closed_image = binary_dilation(bw_image, disk(21))
    closed_image = binary_erosion(closed_image, disk(21))

    if display:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(closed_image, cmap="gray")
        plt.title("Smooth Boundary without Sharp Spikes")
        plt.show()

    # Apply morphological operations to get the external contour
    eroded_image = binary_erosion(closed_image, disk(1))
    external_contour = closed_image & ~eroded_image

    imwrite(imname.with_suffix(".jpg"), img_as_ubyte(external_contour))

    # Dilate the external contour for better visibility
    binary_dilation(external_contour, disk(3))

    # Create a masked image of the same size as the input image
    masked_image = np.zeros_like(img, dtype=np.uint8)
    # Mask out (black) everything outside of closed_image
    masked_image[closed_image] = img[closed_image]

    if display:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(masked_image)
        plt.show()

    return masked_image


class Sequence:
    """Sequence plugin that masks each frame to its largest smooth contour
    before detection and correspondence.

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
                base_image_name = spar.get_img_base_name(i_cam).decode()
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
                base_name = spar.get_img_base_name(i_cam).decode()
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
