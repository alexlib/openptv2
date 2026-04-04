"""Adapter layer to provide Cython-compatible Frame API for Python engine."""

from typing import List, Optional, Union
import numpy as np

from .tracking_frame_buf import Frame as FrameImpl


class Frame:
    """
    Cython-compatible Frame wrapper for Python engine.

    Provides the same API as bindings/optv/tracking_framebuf.pyx Frame class:
    - __init__(num_cams, corres_file_base=None, linkage_file_base=None,
                prio_file_base=None, target_file_base=None, frame_num=None)
    - read(corres_file_base, linkage_file_base, target_file_base, frame_num, prio_file_base)
    - positions() -> (n, 3) numpy array
    - target_positions_for_camera(cam) -> (n, 2) numpy array
    """

    def __init__(
        self,
        num_cams: int,
        corres_file_base: Optional[Union[str, bytes]] = None,
        linkage_file_base: Optional[Union[str, bytes]] = None,
        prio_file_base: Optional[Union[str, bytes]] = None,
        target_file_base: Optional[List[Union[str, bytes]]] = None,
        frame_num: Optional[int] = None,
    ):
        """
        Initialize Frame, optionally loading data from files.

        Arguments:
        num_cams: number of cameras
        corres_file_base: base name for correspondence file
        linkage_file_base: base name for linkage file
        prio_file_base: optional base name for priority file
        target_file_base: list of base names for target files (one per camera)
        frame_num: frame number to append to file names
        """
        self._num_cams = num_cams

        # Convert bytes to str if needed
        if isinstance(corres_file_base, bytes):
            corres_file_base = corres_file_base.decode("utf-8")
        if isinstance(linkage_file_base, bytes):
            linkage_file_base = linkage_file_base.decode("utf-8")
        if isinstance(prio_file_base, bytes):
            prio_file_base = prio_file_base.decode("utf-8")
        if target_file_base is not None:
            target_file_base = [
                tb.decode("utf-8") if isinstance(tb, bytes) else tb
                for tb in target_file_base
            ]

        self._corres_file_base = corres_file_base
        self._linkage_file_base = linkage_file_base
        self._prio_file_base = prio_file_base
        self._target_file_base = target_file_base
        self._frame_num = frame_num

        # Create underlying implementation
        self._impl = FrameImpl(num_cams, max_targets=10000)

        # Auto-load if all parameters provided
        if (
            corres_file_base is not None
            and linkage_file_base is not None
            and target_file_base is not None
            and frame_num is not None
        ):
            self.read(
                corres_file_base,
                linkage_file_base,
                target_file_base,
                frame_num,
                prio_file_base,
            )

    def read(
        self,
        corres_file_base: Union[str, bytes],
        linkage_file_base: Union[str, bytes],
        target_file_base: List[Union[str, bytes]],
        frame_num: int,
        prio_file_base: Optional[Union[str, bytes]] = None,
    ) -> bool:
        """
        Read frame data from files.

        Arguments:
        corres_file_base: base name for correspondence file
        linkage_file_base: base name for linkage file
        target_file_base: list of base names for target files
        frame_num: frame number
        prio_file_base: optional base name for priority file
        """
        # Convert bytes to str if needed
        if isinstance(corres_file_base, bytes):
            corres_file_base = corres_file_base.decode("utf-8")
        if isinstance(linkage_file_base, bytes):
            linkage_file_base = linkage_file_base.decode("utf-8")
        if isinstance(prio_file_base, bytes):
            prio_file_base = prio_file_base.decode("utf-8")
        target_file_base = [
            tb.decode("utf-8") if isinstance(tb, bytes) else tb
            for tb in target_file_base
        ]

        # Cython order: corres, linkage, targets, frame_num, prio
        # Python impl order: corres, linkage, prio, targets, frame_num
        return self._impl.read(
            corres_file_base,
            linkage_file_base,
            prio_file_base if prio_file_base else "",
            target_file_base,
            frame_num,
        )

    def positions(self) -> np.ndarray:
        """
        Returns an (n, 3) array for the 3D positions on n particles in the frame.
        """
        return self._impl.positions()

    def target_positions_for_camera(self, cam: int) -> np.ndarray:
        """
        Gets all targets in this frame as seen by the selected camera.

        Arguments:
        cam: camera number, starting from 0

        Returns:
        (n, 2) array with 2D positions. NaN if no target for that particle.
        """
        return self._impl.target_positions_for_camera(cam)

    @property
    def num_parts(self) -> int:
        """Number of particles in the frame."""
        return self._impl.num_parts

    @property
    def num_cams(self) -> int:
        """Number of cameras."""
        return self._num_cams
