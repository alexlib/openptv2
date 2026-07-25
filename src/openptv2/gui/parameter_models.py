"""Pydantic v2 models for parameter validation."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_validator


class SectionModel(BaseModel):
    """Base for all parameter sections.

    extra="allow" so normalizing a YAML through the model (ParameterManager
    .from_yaml) can never silently drop keys the models don't know about.
    """

    model_config = {"extra": "allow"}


class CalOriParams(SectionModel):
    chfield: int = 0
    fixp_name: str = ""
    img_cal_name: list[str] = []
    img_ori: list[str] = []
    pair_flag: bool = False
    tiff_flag: bool = True
    cal_splitter: bool = False

    @model_validator(mode="after")
    def img_lists_non_empty(self) -> "CalOriParams":
        if not self.img_cal_name:
            raise ValueError("cal_ori.img_cal_name must not be empty")
        if not self.img_ori:
            raise ValueError("cal_ori.img_ori must not be empty")
        if len(self.img_cal_name) != len(self.img_ori):
            raise ValueError("cal_ori.img_cal_name and img_ori must have equal length")
        return self


class CriteriaParams(SectionModel):
    X_lay: list[float] = [-40.0, 40.0]
    Zmax_lay: list[float] = [20.0, 20.0]
    Zmin_lay: list[float] = [-20.0, -20.0]
    cn: float = 0.02
    cnx: float = 0.02
    cny: float = 0.02
    corrmin: float = 33.0
    csumg: float = 0.02
    eps0: float = 0.05


class DetectPlateParams(SectionModel):
    gvth_1: int = 10
    gvth_2: int = 10
    gvth_3: int = 10
    gvth_4: int = 10
    max_npix: int = 400
    max_npix_x: int = 50
    max_npix_y: int = 50
    min_npix: int = 25
    min_npix_x: int = 5
    min_npix_y: int = 5
    size_cross: int = 3
    sum_grey: int = 100
    tol_dis: int = 500


class DumbbellParams(SectionModel):
    dumbbell_eps: float = 3.0
    dumbbell_gradient_descent: float = 0.05
    dumbbell_niter: int = 500
    dumbbell_penalty_weight: float = 1.0
    dumbbell_scale: float = 25.0
    dumbbell_step: int = 1
    dumbbell_fixed_camera: int = 0


class ExamineParams(SectionModel):
    Combine_Flag: bool = False
    Examine_Flag: bool = False


class ManOriParams(SectionModel):
    nr: list[int] = []


class MultiPlanesParams(SectionModel):
    n_planes: int = 1
    plane_name: list[str] = []


class OrientParams(SectionModel):
    cc: int = 0
    interf: int = 0
    k1: int = 0
    k2: int = 0
    k3: int = 0
    p1: int = 0
    p2: int = 0
    pnfo: int = 0
    scale: int = 0
    shear: int = 0
    xh: int = 0
    yh: int = 0


class PftVersionParams(SectionModel):
    Existing_Target: int = 0


class PtvParams(SectionModel):
    allcam_flag: bool = False
    chfield: int = 0
    hp_flag: bool = True
    img_cal: list[str] = []
    img_name: list[str] = []
    imx: int = 1280
    imy: int = 1024
    mmp_d: float = 6.0
    mmp_n1: float = 1.0
    mmp_n2: float = 1.33
    mmp_n3: float = 1.46
    pix_x: float = 0.012
    pix_y: float = 0.012
    tiff_flag: bool = True
    splitter: bool = False
    # Mapping from image quadrant (TL, TR, BL, BR) to camera index when
    # splitter mode is on. The historical hardware default is [0, 1, 3, 2].
    splitter_order: list[int] = [0, 1, 3, 2]


class SequenceParams(SectionModel):
    base_name: list[str] = []
    first: int = 0
    last: int = 0


class ShakingParams(SectionModel):
    shaking_first_frame: int = 0
    shaking_last_frame: int = 0
    shaking_max_num_frames: int = 5
    shaking_max_num_points: int = 10


class SortGridParams(SectionModel):
    radius: int = 20


class TargRecParams(SectionModel):
    cr_sz: int = 2
    disco: int = 100
    gvthres: list[int] = []
    nnmax: int = 500
    nnmin: int = 4
    nxmax: int = 100
    nxmin: int = 2
    nymax: int = 100
    nymin: int = 2
    sumg_min: int = 150


class TrackParams(SectionModel):
    preset: str = "full_multipass"
    angle: float = 120.0
    dacc: float = 5.5
    dvxmax: float = 15.5
    dvxmin: float = -15.5
    dvymax: float = 15.5
    dvymin: float = -15.5
    dvzmax: float = 15.5
    dvzmin: float = -15.5
    flagNewParticles: bool = True
    track_mode: int = 0
    postprocess: bool = True


class MaskingParams(SectionModel):
    mask_flag: bool = False
    mask_base_name: str = ""


class UnsharpMaskParams(SectionModel):
    flag: bool = False
    size: int = 3
    strength: float = 1.0


class PluginsParams(SectionModel):
    available_tracking: list[str] = ["default"]
    available_sequence: list[str] = ["default"]
    selected_tracking: str = "default"
    selected_sequence: str = "default"


class ManOriPoint(SectionModel):
    x: float
    y: float


class ManOriCamera(SectionModel):
    point_1: ManOriPoint = ManOriPoint(x=0.0, y=0.0)
    point_2: ManOriPoint = ManOriPoint(x=0.0, y=0.0)
    point_3: ManOriPoint = ManOriPoint(x=0.0, y=0.0)
    point_4: ManOriPoint = ManOriPoint(x=0.0, y=0.0)


class AllParams(SectionModel):
    num_cams: int
    cal_ori: CalOriParams
    criteria: CriteriaParams = CriteriaParams()
    detect_plate: DetectPlateParams = DetectPlateParams()
    dumbbell: DumbbellParams = DumbbellParams()
    examine: ExamineParams = ExamineParams()
    man_ori: ManOriParams = ManOriParams()
    multi_planes: MultiPlanesParams = MultiPlanesParams()
    orient: OrientParams = OrientParams()
    pft_version: PftVersionParams = PftVersionParams()
    ptv: PtvParams
    sequence: SequenceParams = SequenceParams()
    shaking: ShakingParams = ShakingParams()
    sortgrid: SortGridParams = SortGridParams()
    targ_rec: TargRecParams = TargRecParams()
    track: TrackParams = TrackParams()
    masking: MaskingParams = MaskingParams()
    unsharp_mask: UnsharpMaskParams = UnsharpMaskParams()
    plugins: PluginsParams = PluginsParams()
    man_ori_coordinates: dict[str, Any] = {}

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def cam_list_lengths(self) -> "AllParams":
        n = self.num_cams
        if n and self.ptv.img_name and len(self.ptv.img_name) != n:
            raise ValueError(f"ptv.img_name length {len(self.ptv.img_name)} != num_cams {n}")
        if n and self.cal_ori.img_cal_name and len(self.cal_ori.img_cal_name) != n:
            raise ValueError(
                f"cal_ori.img_cal_name length {len(self.cal_ori.img_cal_name)} != num_cams {n}"
            )
        return self
