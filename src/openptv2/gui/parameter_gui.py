from traits.api import Bool, Enum, Float, HasTraits, Int, List, Str
from traitsui.api import (
    EnumEditor,
    Group,
    Handler,
    HGroup,
    Item,
    Tabbed,
    VGroup,
    View,
    spring,
)

from openptv2.tracking_presets import (
    PRESET_CHOICES,
    PRESET_CONFIGS,
    apply_preset,
    infer_preset,
)

from .experiment import Experiment

DEFAULT_STRING = "---"
DEFAULT_INT = -999
DEFAULT_FLOAT = -999.0

DEFAULT_SPLITTER_ORDER = [0, 1, 3, 2]


def parse_splitter_order(text) -> list:
    """Parse a '0,1,3,2'-style string into a permutation of [0, 1, 2, 3]."""
    try:
        order = [int(v) for v in str(text).replace(",", " ").split()]
    except ValueError:
        order = []
    if sorted(order) != [0, 1, 2, 3]:
        print(
            f"Warning: invalid splitter order {text!r}; "
            f"using default {DEFAULT_SPLITTER_ORDER}"
        )
        return list(DEFAULT_SPLITTER_ORDER)
    return order


# define handler function for main parameters
class ParamHandler(Handler):
    def closed(self, info, is_ok):
        if is_ok:
            main_params = info.object
            experiment = main_params.experiment

            print("Updating parameters via Experiment...")

            # Update top-level num_cams
            experiment.pm.parameters["num_cams"] = main_params.Num_Cam

            # Update ptv.par
            img_name = [
                main_params.Name_1_Image,
                main_params.Name_2_Image,
                main_params.Name_3_Image,
                main_params.Name_4_Image,
            ]
            img_cal_name = [
                main_params.Cali_1_Image,
                main_params.Cali_2_Image,
                main_params.Cali_3_Image,
                main_params.Cali_4_Image,
            ]

            img_name = img_name[: main_params.Num_Cam]
            img_cal_name = img_cal_name[: main_params.Num_Cam]

            experiment.pm.parameters["ptv"].update(
                {
                    "img_name": img_name,
                    "img_cal": img_cal_name,
                    "hp_flag": main_params.HighPass,
                    "allcam_flag": main_params.Accept_OnlyAllCameras,
                    "tiff_flag": main_params.tiff_flag,
                    "imx": main_params.imx,
                    "imy": main_params.imy,
                    "pix_x": main_params.pix_x,
                    "pix_y": main_params.pix_y,
                    "chfield": main_params.chfield,
                    "mmp_n1": main_params.Refr_Air,
                    "mmp_n2": main_params.Refr_Glass,
                    "mmp_n3": main_params.Refr_Water,
                    "mmp_d": main_params.Thick_Glass,
                    "splitter": main_params.Splitter,
                    "splitter_order": parse_splitter_order(main_params.Splitter_Order),
                    "negative": main_params.Negative,
                    "parallel_preprocess": main_params.Parallel_Preprocess,
                    "num_workers": main_params.Num_Workers,
                }
            )

            # Update cal_ori.par
            # experiment.pm.parameters['cal_ori'].update({
            #     'fixp_name': main_params.fixp_name,
            #     'img_cal_name': main_params.img_cal_name, 'img_ori': main_params.img_ori,
            #     'tiff_flag': main_params.tiff_flag, 'pair_flag': main_params.pair_Flag,
            #     'chfield': main_params.chfield
            # })

            # Update targ_rec.par
            gvthres = [
                main_params.Gray_Tresh_1,
                main_params.Gray_Tresh_2,
                main_params.Gray_Tresh_3,
                main_params.Gray_Tresh_4,
            ]
            gvthres = gvthres[: main_params.Num_Cam]

            experiment.pm.parameters["targ_rec"].update(
                {
                    "gvthres": gvthres,
                    "disco": main_params.Tol_Disc,
                    "nnmin": main_params.Min_Npix,
                    "nnmax": main_params.Max_Npix,
                    "nxmin": main_params.Min_Npix_x,
                    "nxmax": main_params.Max_Npix_x,
                    "nymin": main_params.Min_Npix_y,
                    "nymax": main_params.Max_Npix_y,
                    "sumg_min": main_params.Sum_Grey,
                    "cr_sz": main_params.Size_Cross,
                }
            )

            # Update pft_version.par
            if "pft_version" not in experiment.pm.parameters:
                experiment.pm.parameters["pft_version"] = {}
            experiment.pm.parameters["pft_version"]["Existing_Target"] = int(
                main_params.Existing_Target
            )

            # Update sequence.par
            base_name = [
                main_params.Basename_1_Seq,
                main_params.Basename_2_Seq,
                main_params.Basename_3_Seq,
                main_params.Basename_4_Seq,
            ]
            base_name = base_name[: main_params.Num_Cam]

            experiment.pm.parameters["sequence"].update(
                {
                    "base_name": base_name,
                    "first": main_params.Seq_First,
                    "last": main_params.Seq_Last,
                }
            )

            # Update criteria.par
            X_lay = [main_params.Xmin, main_params.Xmax]
            Zmin_lay = [main_params.Zmin1, main_params.Zmin2]
            Zmax_lay = [main_params.Zmax1, main_params.Zmax2]
            experiment.pm.parameters["criteria"].update(
                {
                    "X_lay": X_lay,
                    "Zmin_lay": Zmin_lay,
                    "Zmax_lay": Zmax_lay,
                    "cnx": main_params.Min_Corr_nx,
                    "cny": main_params.Min_Corr_ny,
                    "cn": main_params.Min_Corr_npix,
                    "csumg": main_params.Sum_gv,
                    "corrmin": main_params.Min_Weight_corr,
                    "eps0": main_params.Tol_Band,
                }
            )

            # Update masking parameters
            if "masking" not in experiment.pm.parameters:
                experiment.pm.parameters["masking"] = {}
            experiment.pm.parameters["masking"].update(
                {
                    "mask_flag": main_params.Subtr_Mask,
                    "mask_base_name": main_params.Base_Name_Mask,
                }
            )

            # Save all changes to the YAML file through the experiment
            experiment.save_active()
            print("Parameters saved successfully!")
            if main_params.Negative:
                print(
                    "[WARNING] You must reload images for 'Negative images?' to take effect."
                )


# define handler function for calibration parameters
class CalHandler(Handler):
    def closed(self, info, is_ok):
        if is_ok:
            calib_params = info.object
            experiment = calib_params.experiment
            num_cams = experiment.pm.parameters["num_cams"]

            print("Updating calibration parameters via Experiment...")

            # Update top-level num_cams
            # experiment.pm.parameters['num_cams'] = calib_params.n_img

            # Update ptv.par with some parameters that for some reason
            # are stored in Calibration Parameters GUI
            experiment.pm.parameters["ptv"].update(
                {
                    # 'tiff_flag': calib_params.tiff_head,
                    "imx": calib_params.h_image_size,
                    "imy": calib_params.v_image_size,
                    "pix_x": calib_params.h_pixel_size,
                    "pix_y": calib_params.v_pixel_size,
                    # 'chfield': calib_params.chfield,
                }
            )

            # Update cal_ori.par
            img_cal_name = [
                calib_params.cam_1,
                calib_params.cam_2,
                calib_params.cam_3,
                calib_params.cam_4,
            ]
            img_ori = [
                calib_params.ori_cam_1,
                calib_params.ori_cam_2,
                calib_params.ori_cam_3,
                calib_params.ori_cam_4,
            ]

            img_cal_name = img_cal_name[:num_cams]
            img_ori = img_ori[:num_cams]

            experiment.pm.parameters["cal_ori"].update(
                {
                    "fixp_name": calib_params.fixp_name,
                    "img_cal_name": img_cal_name,  # see above
                    "img_ori": img_ori,  # see above
                    #'tiff_flag': calib_params.tiff_head,
                    #'pair_flag': calib_params.pair_head,
                    #'chfield': calib_params.chfield,
                    "cal_splitter": calib_params._cal_splitter,
                }
            )

            # Update detect_plate.par
            if "detect_plate" not in experiment.pm.parameters:
                experiment.pm.parameters["detect_plate"] = {}
            experiment.pm.parameters["detect_plate"].update(
                {
                    "gvth_1": calib_params.grey_value_treshold_1,
                    "gvth_2": calib_params.grey_value_treshold_2,
                    "gvth_3": calib_params.grey_value_treshold_3,
                    "gvth_4": calib_params.grey_value_treshold_4,
                    "tol_dis": calib_params.tolerable_discontinuity,
                    "min_npix": calib_params.min_npix,
                    "max_npix": calib_params.max_npix,
                    "min_npix_x": calib_params.min_npix_x,
                    "max_npix_x": calib_params.max_npix_x,
                    "min_npix_y": calib_params.min_npix_y,
                    "max_npix_y": calib_params.max_npix_y,
                    "sum_grey": calib_params.sum_of_grey,
                    "size_cross": calib_params.size_of_crosses,
                }
            )

            # Update ONLY the 'man_ori' section (legacy indices), never touch 'man_ori_coordinates' here
            nr1 = [
                calib_params.img_1_p1,
                calib_params.img_1_p2,
                calib_params.img_1_p3,
                calib_params.img_1_p4,
            ]
            nr2 = [
                calib_params.img_2_p1,
                calib_params.img_2_p2,
                calib_params.img_2_p3,
                calib_params.img_2_p4,
            ]
            nr3 = [
                calib_params.img_3_p1,
                calib_params.img_3_p2,
                calib_params.img_3_p3,
                calib_params.img_3_p4,
            ]
            nr4 = [
                calib_params.img_4_p1,
                calib_params.img_4_p2,
                calib_params.img_4_p3,
                calib_params.img_4_p4,
            ]
            nr = nr1 + nr2 + nr3 + nr4
            if "man_ori" not in experiment.pm.parameters:
                experiment.pm.parameters["man_ori"] = {}
            experiment.pm.parameters["man_ori"]["nr"] = nr
            # Do NOT update or remove 'man_ori_coordinates' here; that is managed only by calibration_gui

            # Update examine.par
            if "examine" not in experiment.pm.parameters:
                experiment.pm.parameters["examine"] = {}
            experiment.pm.parameters["examine"]["Examine_Flag"] = (
                calib_params.Examine_Flag
            )
            experiment.pm.parameters["examine"]["Combine_Flag"] = (
                calib_params.Combine_Flag
            )

            # Update orient.par
            if "orient" not in experiment.pm.parameters:
                experiment.pm.parameters["orient"] = {}
            experiment.pm.parameters["orient"].update(
                {
                    "pnfo": calib_params.point_number_of_orientation,
                    "cc": int(calib_params.cc),
                    "xh": int(calib_params.xh),
                    "yh": int(calib_params.yh),
                    "k1": int(calib_params.k1),
                    "k2": int(calib_params.k2),
                    "k3": int(calib_params.k3),
                    "p1": int(calib_params.p1),
                    "p2": int(calib_params.p2),
                    "scale": int(calib_params.scale),
                    "shear": int(calib_params.shear),
                    "interf": int(calib_params.interf),
                }
            )

            # Update shaking.par
            if "shaking" not in experiment.pm.parameters:
                experiment.pm.parameters["shaking"] = {}
            experiment.pm.parameters["shaking"].update(
                {
                    "shaking_first_frame": calib_params.shaking_first_frame,
                    "shaking_last_frame": calib_params.shaking_last_frame,
                    "shaking_max_num_points": calib_params.shaking_max_num_points,
                    "shaking_max_num_frames": calib_params.shaking_max_num_frames,
                    "shaking_tol_px": calib_params.shaking_tol_px,
                    "shaking_hold_cam": calib_params.shaking_hold_cam,
                }
            )

            # Update dumbbell.par
            if "dumbbell" not in experiment.pm.parameters:
                experiment.pm.parameters["dumbbell"] = {}
            experiment.pm.parameters["dumbbell"].update(
                {
                    "dumbbell_eps": calib_params.dumbbell_eps,
                    "dumbbell_scale": calib_params.dumbbell_scale,
                    "dumbbell_gradient_descent": calib_params.dumbbell_gradient_descent,
                    "dumbbell_penalty_weight": calib_params.dumbbell_penalty_weight,
                    "dumbbell_step": calib_params.dumbbell_step,
                    "dumbbell_niter": calib_params.dumbbell_niter,
                    "dumbbell_fixed_camera": calib_params.dumbbell_fixed_camera,
                }
            )

            # Save all changes to the YAML file through the experiment
            experiment.save_active()
            print("Calibration parameters saved successfully!")


class TrackHandler(Handler):
    def closed(self, info, is_ok):
        if is_ok:
            track_params = info.object
            experiment = track_params.experiment

            print("Updating tracking parameters via Experiment...")

            if "track" not in experiment.pm.parameters:
                experiment.pm.parameters["track"] = {}
            if "plugins" not in experiment.pm.parameters:
                experiment.pm.parameters["plugins"] = {}

            preset_key = track_params.preset
            existing_plugin = experiment.pm.parameters.get("plugins", {}).get(
                "selected_tracking"
            )

            t_dict, p_dict = apply_preset(
                preset_key,
                {
                    "dvxmin": track_params.dvxmin,
                    "dvxmax": track_params.dvxmax,
                    "dvymin": track_params.dvymin,
                    "dvymax": track_params.dvymax,
                    "dvzmin": track_params.dvzmin,
                    "dvzmax": track_params.dvzmax,
                    "angle": track_params.angle,
                    "dacc": track_params.dacc,
                    "flagNewParticles": track_params.flagNewParticles,
                    "track_mode": int(track_params.track_mode),
                    "postprocess": bool(track_params.postprocess),
                },
                experiment.pm.parameters.get("plugins", {}),
                custom_plugin_name=existing_plugin
                if preset_key == "custom_plugin"
                else None,
            )

            experiment.pm.parameters["track"].update(t_dict)
            if p_dict:
                experiment.pm.parameters["plugins"].update(p_dict)

            experiment.save_active()
            print("Tracking parameters saved successfully!")


class Tracking_Params(HasTraits):
    preset = Enum(*[key for key, label in PRESET_CHOICES])
    active_plugin = Str("default")
    dvxmin = Float()
    dvxmax = Float()
    dvymin = Float()
    dvymax = Float()
    dvzmin = Float()
    dvzmax = Float()
    angle = Float()
    dacc = Float()
    flagNewParticles = Bool(True)
    track_mode = Enum(0, 1)
    postprocess = Bool(True)

    def __init__(self, experiment: Experiment):
        super(Tracking_Params, self).__init__()
        self.experiment = experiment
        tracking_params = experiment.pm.get_section("track")
        plugins_params = experiment.pm.parameters.get("plugins", {})

        self.dvxmin = float(tracking_params.get("dvxmin", -10.0))
        self.dvxmax = float(tracking_params.get("dvxmax", 10.0))
        self.dvymin = float(tracking_params.get("dvymin", -10.0))
        self.dvymax = float(tracking_params.get("dvymax", 10.0))
        self.dvzmin = float(tracking_params.get("dvzmin", -10.0))
        self.dvzmax = float(tracking_params.get("dvzmax", 10.0))
        self.angle = float(tracking_params.get("angle", 120.0))
        self.dacc = float(tracking_params.get("dacc", 5.0))
        self.flagNewParticles = bool(tracking_params.get("flagNewParticles", True))
        self.track_mode = int(tracking_params.get("track_mode", 0))
        self.postprocess = bool(tracking_params.get("postprocess", True))

        self.active_plugin = str(plugins_params.get("selected_tracking", "default"))
        self.preset = infer_preset(tracking_params, plugins_params)

    def _preset_changed(self, old, new):
        if new in PRESET_CONFIGS:
            cfg = PRESET_CONFIGS[new]
            self.track_mode = cfg["track_mode"]
            self.flagNewParticles = cfg["flagNewParticles"]
            self.postprocess = cfg["postprocess"]

    Tracking_Params_View = View(
        VGroup(
            Item(
                name="preset",
                label="Tracking Strategy / Preset:",
                editor=EnumEditor(values={key: label for key, label in PRESET_CHOICES}),
            ),
            Item(
                name="active_plugin",
                style="readonly",
                label="Active Selected Plugin:",
            ),
            show_border=True,
            label="Pipeline Strategy",
        ),
        VGroup(
            HGroup(
                Item(name="dvxmin", label="dvxmin:"),
                Item(name="dvxmax", label="dvxmax:"),
            ),
            HGroup(
                Item(name="dvymin", label="dvymin:"),
                Item(name="dvymax", label="dvymax:"),
            ),
            HGroup(
                Item(name="dvzmin", label="dvzmin:"),
                Item(name="dvzmax", label="dvzmax:"),
            ),
            HGroup(
                Item(name="angle", label="Angle [gon]:"),
                Item(name="dacc", label="dacc:"),
            ),
            show_border=True,
            label="Search Box & Kinematic Limits",
        ),
        buttons=["Undo", "OK", "Cancel"],
        handler=TrackHandler(),
        title="Tracking Parameters",
    )


class Main_Params(HasTraits):
    # Panel 1: General
    Num_Cam = Int(label="Number of cameras: ")
    Accept_OnlyAllCameras = Bool(label="Accept only points seen from all cameras?")
    pair_Flag = Bool(label="Include pairs")
    pair_enable_flag = True
    all_enable_flag = False
    # hp_enable_flag = Bool()
    negative_image_flag = Bool()
    Splitter = Bool(label="Split images into 4?")
    Splitter_Order = Str("0,1,3,2", label="Splitter view order (TL,TR,BL,BR)")

    tiff_flag = Bool()
    imx = Int()
    imy = Int()
    pix_x = Float()
    pix_y = Float()
    chfield = Int()
    img_cal_name = List()

    fixp_name = Str()
    img_ori = List()

    Name_1_Image = Str(label="Name of 1. image")
    Name_2_Image = Str(label="Name of 2. image")
    Name_3_Image = Str(label="Name of 3. image")
    Name_4_Image = Str(label="Name of 4. image")
    Cali_1_Image = Str(label="Calibration data for 1. image")
    Cali_2_Image = Str(label="Calibration data for 2. image")
    Cali_3_Image = Str(label="Calibration data for 3. image")
    Cali_4_Image = Str(label="Calibration data for 4. image")

    Refr_Air = Float(label="Air:")
    Refr_Glass = Float(label="Glass:")
    Refr_Water = Float(label="Water:")
    Thick_Glass = Float(label="Thickness of glass:")

    # New panel 2: ImageProcessing
    HighPass = Bool(label="High pass filter")
    Gray_Tresh_1 = Int(label="1st image")
    Gray_Tresh_2 = Int(label="2nd image")
    Gray_Tresh_3 = Int(label="3rd image")
    Gray_Tresh_4 = Int(label="4th image")
    Min_Npix = Int(label="min npix")
    Max_Npix = Int(label="max npix")
    Min_Npix_x = Int(label="min npix x")
    Max_Npix_x = Int(label="max npix x")
    Min_Npix_y = Int(label="min npix y")
    Max_Npix_y = Int(label="max npix y")
    Sum_Grey = Int(label="Sum of grey value")
    Tol_Disc = Int(label="Tolerable discontinuity")
    Size_Cross = Int(label="Size of crosses")
    Subtr_Mask = Bool(label="Subtract mask")
    Base_Name_Mask = Str(label="Base name for the mask")
    Existing_Target = Bool(label="Use existing_target files?")
    Negative = Bool(label="Negative images?")

    # New panel 3: Sequence
    Seq_First = Int(label="First sequence image:")
    Seq_Last = Int(label="Last sequence image:")
    Basename_1_Seq = Str(label="Basename for 1. sequence")
    Basename_2_Seq = Str(label="Basename for 2. sequence")
    Basename_3_Seq = Str(label="Basename for 3. sequence")
    Basename_4_Seq = Str(label="Basename for 4. sequence")
    Parallel_Preprocess = Bool(label="Parallel Pre-processing")
    Num_Workers = Int(label="Number of workers (0 for auto)")

    # Panel 4: ObservationVolume
    Xmin = Float(label="Xmin")
    Xmax = Float(label="Xmax")
    Zmin1 = Float(label="Zmin")
    Zmin2 = Float(label="Zmin")
    Zmax1 = Float(label="Zmax")
    Zmax2 = Float(label="Zmax")

    # Panel 5: ParticleDetection
    Min_Corr_nx = Float(label="min corr for ratio nx")
    Min_Corr_ny = Float(label="min corr for ratio ny")
    Min_Corr_npix = Float(label="min corr for ratio npix")
    Sum_gv = Float(label="sum of gv")
    Min_Weight_corr = Float(label="min for weighted correlation")
    Tol_Band = Float(lable="Tolerance of epipolar band [mm]")

    Group1 = Group(
        Group(
            Item(name="Num_Cam", width=30),
            Item(name="Splitter"),
            Item(name="Splitter_Order", enabled_when="Splitter"),
            Item(name="Accept_OnlyAllCameras", enabled_when="all_enable_flag"),
            Item(name="pair_Flag", enabled_when="pair_enable_flag"),
            orientation="horizontal",
        ),
        Group(
            Group(
                Item(name="Name_1_Image", width=150),
                Item(name="Name_2_Image"),
                Item(name="Name_3_Image"),
                Item(name="Name_4_Image"),
                orientation="vertical",
            ),
            Group(
                Item(name="Cali_1_Image", width=150),
                Item(name="Cali_2_Image"),
                Item(name="Cali_3_Image"),
                Item(name="Cali_4_Image"),
                orientation="vertical",
            ),
            orientation="horizontal",
        ),
        orientation="vertical",
        label="General",
    )

    Group2 = Group(
        Group(
            Item(name="Refr_Air"),
            Item(name="Refr_Glass"),
            Item(name="Refr_Water"),
            Item(name="Thick_Glass"),
            orientation="horizontal",
        ),
        label="Refractive Indices",
        show_border=True,
        orientation="vertical",
    )

    Group3 = Group(
        Group(
            Item(name="Gray_Tresh_1"),
            Item(name="Gray_Tresh_2"),
            Item(name="Gray_Tresh_3"),
            Item(name="Gray_Tresh_4"),
            label="Gray value treshold: ",
            show_border=True,
            orientation="horizontal",
        ),
        Group(
            Group(
                Item(name="Min_Npix"),
                Item(name="Max_Npix"),
                Item(name="Sum_Grey"),
                orientation="vertical",
            ),
            Group(
                Item(name="Min_Npix_x"),
                Item(name="Max_Npix_x"),
                Item(name="Tol_Disc"),
                orientation="vertical",
            ),
            Group(
                Item(name="Min_Npix_y"),
                Item(name="Max_Npix_y"),
                Item(name="Size_Cross"),
                orientation="vertical",
            ),
            orientation="horizontal",
        ),
        Group(
            Item(name="Subtr_Mask"),
            Item(name="Base_Name_Mask"),
            Item(name="Existing_Target"),
            Item(name="HighPass"),
            Item(name="Negative"),
            orientation="horizontal",
        ),
        orientation="vertical",
        show_border=True,
        label="Particle recognition",
    )

    Group4 = Group(
        Group(
            Item(name="Seq_First"),
            Item(name="Seq_Last"),
            orientation="horizontal",
        ),
        Group(
            Item(name="Basename_1_Seq"),
            Item(name="Basename_2_Seq"),
            Item(name="Basename_3_Seq"),
            Item(name="Basename_4_Seq"),
            orientation="vertical",
        ),
        Group(
            Item(name="Parallel_Preprocess"),
            Item(name="Num_Workers"),
            orientation="horizontal",
        ),
        label="Parameters for sequence processing",
        orientation="vertical",
        show_border=True,
    )

    Group5 = Group(
        Group(Item(name="Xmin"), Item(name="Xmax"), orientation="vertical"),
        Group(Item(name="Zmin1"), Item(name="Zmin2"), orientation="vertical"),
        Group(Item(name="Zmax1"), Item(name="Zmax2"), orientation="vertical"),
        orientation="horizontal",
        label="Observation Volume",
        show_border=True,
    )

    Group6 = Group(
        Group(
            Item(name="Min_Corr_nx"),
            Item(name="Min_Corr_npix"),
            Item(name="Min_Weight_corr"),
            orientation="vertical",
        ),
        Group(
            Item(name="Min_Corr_ny"),
            Item(name="Sum_gv"),
            Item(name="Tol_Band"),
            orientation="vertical",
        ),
        orientation="horizontal",
        label="Criteria for correspondences",
        show_border=True,
    )

    Main_Params_View = View(
        Tabbed(Group1, Group2, Group3, Group4, Group5, Group6),
        resizable=True,
        width=0.5,
        height=0.3,
        dock="horizontal",
        buttons=["Undo", "OK", "Cancel"],
        handler=ParamHandler(),
        title="Main Parameters",
    )

    def _pair_Flag_fired(self):
        if self.pair_Flag:
            self.all_enable_flag = False
        else:
            self.all_enable_flag = True

    def _Accept_OnlyAllCameras_fired(self):
        if self.Accept_OnlyAllCameras:
            self.pair_enable_flag = False
        else:
            self.pair_enable_flag = True

    def _reload(self, num_cams: int, pm):
        global_n_cam = num_cams
        ptv_params = pm.get_section("ptv")

        img_names = ptv_params["img_name"]
        for i, name in enumerate(img_names):
            if name is not None and i < global_n_cam:
                setattr(self, f"Name_{i + 1}_Image", name)

        img_cals = ptv_params["img_cal"]
        for i, cal in enumerate(img_cals):
            if cal is not None and i < global_n_cam:
                setattr(self, f"Cali_{i + 1}_Image", cal)

        self.Refr_Air = ptv_params["mmp_n1"]
        self.Refr_Glass = ptv_params["mmp_n2"]
        self.Refr_Water = ptv_params["mmp_n3"]
        self.Thick_Glass = ptv_params["mmp_d"]
        self.Accept_OnlyAllCameras = bool(ptv_params["allcam_flag"])
        self.Num_Cam = global_n_cam
        self.HighPass = bool(ptv_params["hp_flag"])
        self.tiff_flag = bool(ptv_params["tiff_flag"])
        self.imx = int(ptv_params["imx"])
        self.imy = int(ptv_params["imy"])
        self.pix_x = ptv_params["pix_x"]
        self.pix_y = ptv_params["pix_y"]
        self.chfield = int(ptv_params["chfield"])
        self.Negative = bool(ptv_params.get("negative", False))
        self.Splitter = bool(ptv_params.get("splitter", False))
        self.Splitter_Order = ",".join(
            str(v) for v in ptv_params.get("splitter_order", DEFAULT_SPLITTER_ORDER)
        )

        targ_rec_params = pm.get_section("targ_rec")
        gvthres = targ_rec_params["gvthres"]
        for i in range(num_cams):
            if i < len(gvthres):
                setattr(self, f"Gray_Tresh_{i + 1}", int(gvthres[i]))

        self.Min_Npix = int(targ_rec_params["nnmin"])
        self.Max_Npix = int(targ_rec_params["nnmax"])
        self.Min_Npix_x = int(targ_rec_params["nxmin"])
        self.Max_Npix_x = int(targ_rec_params["nxmax"])
        self.Min_Npix_y = int(targ_rec_params["nymin"])
        self.Max_Npix_y = int(targ_rec_params["nymax"])
        self.Sum_Grey = int(targ_rec_params["sumg_min"])
        self.Tol_Disc = int(targ_rec_params["disco"])
        self.Size_Cross = int(targ_rec_params["cr_sz"])

        pft_version_params = pm.get_section("pft_version")
        self.Existing_Target = bool(pft_version_params["Existing_Target"])

        sequence_params = pm.get_section("sequence")
        base_names = sequence_params["base_name"]
        for i, base_name in enumerate(base_names):
            if base_name is not None and i < global_n_cam:
                setattr(self, f"Basename_{i + 1}_Seq", base_name)

        self.Seq_First = int(sequence_params["first"])
        self.Seq_Last = int(sequence_params["last"])

        self.Parallel_Preprocess = bool(ptv_params.get("parallel_preprocess", False))
        self.Num_Workers = int(ptv_params.get("num_workers", 0))

        criteria_params = pm.get_section("criteria")
        X_lay = criteria_params["X_lay"]
        self.Xmin, self.Xmax = X_lay[:2]
        Zmin_lay = criteria_params["Zmin_lay"]
        self.Zmin1, self.Zmin2 = Zmin_lay[:2]
        Zmax_lay = criteria_params["Zmax_lay"]
        self.Zmax1, self.Zmax2 = Zmax_lay[:2]
        self.Min_Corr_nx = criteria_params["cnx"]
        self.Min_Corr_ny = criteria_params["cny"]
        self.Min_Corr_npix = criteria_params["cn"]
        self.Sum_gv = criteria_params["csumg"]
        self.Min_Weight_corr = criteria_params["corrmin"]
        self.Tol_Band = criteria_params["eps0"]

        masking_params = pm.get_section("masking")
        self.Subtr_Mask = masking_params["mask_flag"]
        self.Base_Name_Mask = masking_params["mask_base_name"]

    def __init__(self, experiment: Experiment):
        HasTraits.__init__(self)
        self.experiment = experiment
        self._reload(experiment.get_n_cam(), experiment.pm)


# -----------------------------------------------------------------------------
class Calib_Params(HasTraits):
    # general and unsed variables
    # pair_enable_flag = Bool(True)
    num_cams = Int
    img_name = List
    img_cal = List
    hp_flag = Bool(label="highpass")
    # allcam_flag = Bool(False, label="all camera targets")
    mmp_n1 = Float
    mmp_n2 = Float
    mmp_n3 = Float
    mmp_d = Float
    _cal_splitter = Bool(label="Split calibration image into 4?")

    # images data
    cam_1 = Str(label="Calibration picture camera 1")
    cam_2 = Str(label="Calibration picture camera 2")
    cam_3 = Str(label="Calibration picture camera 3")
    cam_4 = Str(label="Calibration picture camera 4")
    ori_cam_1 = Str(label="Orientation data picture camera 1")
    ori_cam_2 = Str(label="Orientation data picture camera 2")
    ori_cam_3 = Str(label="Orientation data picture camera 3")
    ori_cam_4 = Str(label="Orientation data picture camera 4")

    fixp_name = Str(label="File of Coordinates on plate")
    # tiff_head = Bool(True, label="TIFF-Header")
    # pair_head = Bool(True, label="Include pairs")
    # chfield = Enum("Frame", "Field odd", "Field even")

    Group1_1 = Group(
        Item(name="cam_1"),
        Item(name="cam_2"),
        Item(name="cam_3"),
        Item(name="cam_4"),
        label="Calibration images",
        show_border=True,
    )
    Group1_2 = Group(
        Item(name="ori_cam_1"),
        Item(name="ori_cam_2"),
        Item(name="ori_cam_3"),
        Item(name="ori_cam_4"),
        label="Orientation data",
        show_border=True,
    )
    Group1_3 = Group(
        Item(name="fixp_name"),
        # Group(
        #     # Item(name="tiff_head"),
        #     # Item(name="pair_head", enabled_when="pair_enable_flag"),
        #     # Item(name="chfield", show_label=False, style="custom"),
        #     orientation="vertical",
        #     columns=3,
        # ),
        orientation="vertical",
    )

    Group1 = Group(
        Group1_1,
        Group1_2,
        Group1_3,
        orientation="vertical",
        label="Images Data",
    )

    # calibration data detection

    h_image_size = Int(label="Image size horizontal")
    v_image_size = Int(label="Image size vertical")
    h_pixel_size = Float(label="Pixel size horizontal")
    v_pixel_size = Float(label="Pixel size vertical")

    grey_value_treshold_1 = Int(label="First Image")
    grey_value_treshold_2 = Int(label="Second Image")
    grey_value_treshold_3 = Int(label="Third Image")
    grey_value_treshold_4 = Int(label="Forth Image")
    tolerable_discontinuity = Int(label="Tolerable discontinuity")
    min_npix = Int(label="min npix")
    min_npix_x = Int(label="min npix in x")
    min_npix_y = Int(label="min npix in y")
    max_npix = Int(label="max npix")
    max_npix_x = Int(label="max npix in x")
    max_npix_y = Int(label="max npix in y")
    sum_of_grey = Int(label="Sum of greyvalue")
    size_of_crosses = Int(label="Size of crosses")

    Group2_1 = Group(
        Item(name="h_image_size"),
        Item(name="v_image_size"),
        Item(name="h_pixel_size"),
        Item(name="v_pixel_size"),
        label="Image properties",
        show_border=True,
        orientation="horizontal",
    )

    Group2_2 = (
        Group(
            Item(name="grey_value_treshold_1"),
            Item(name="grey_value_treshold_2"),
            Item(name="grey_value_treshold_3"),
            Item(name="grey_value_treshold_4"),
            orientation="horizontal",
            label="Grayvalue threshold",
            show_border=True,
        ),
    )

    Group2_3 = Group(
        Group(
            Item(name="min_npix"),
            Item(name="min_npix_x"),
            Item(name="min_npix_y"),
            orientation="vertical",
        ),
        Group(
            Item(name="max_npix"),
            Item(name="max_npix_x"),
            Item(name="max_npix_y"),
            orientation="vertical",
        ),
        Group(
            Item(name="tolerable_discontinuity"),
            Item(name="sum_of_grey"),
            Item(name="size_of_crosses"),
            orientation="vertical",
        ),
        orientation="horizontal",
    )

    Group2 = Group(
        Group2_1,
        Group2_2,
        Group2_3,
        orientation="vertical",
        label="Calibration Data Detection",
    )

    # manuel pre orientation
    img_1_p1 = Int(label="P1")
    img_1_p2 = Int(label="P2")
    img_1_p3 = Int(label="P3")
    img_1_p4 = Int(label="P4")
    img_2_p1 = Int(label="P1")
    img_2_p2 = Int(label="P2")
    img_2_p3 = Int(label="P3")
    img_2_p4 = Int(label="P4")
    img_3_p1 = Int(label="P1")
    img_3_p2 = Int(label="P2")
    img_3_p3 = Int(label="P3")
    img_3_p4 = Int(label="P4")
    img_4_p1 = Int(label="P1")
    img_4_p2 = Int(label="P2")
    img_4_p3 = Int(label="P3")
    img_4_p4 = Int(label="P4")

    Group3_1 = Group(
        Item(name="img_1_p1"),
        Item(name="img_1_p2"),
        Item(name="img_1_p3"),
        Item(name="img_1_p4"),
        orientation="horizontal",
        label="Image 1",
        show_border=True,
    )
    Group3_2 = Group(
        Item(name="img_2_p1"),
        Item(name="img_2_p2"),
        Item(name="img_2_p3"),
        Item(name="img_2_p4"),
        orientation="horizontal",
        label="Image 2",
        show_border=True,
    )
    Group3_3 = Group(
        Item(name="img_3_p1"),
        Item(name="img_3_p2"),
        Item(name="img_3_p3"),
        Item(name="img_3_p4"),
        orientation="horizontal",
        label="Image 3",
        show_border=True,
    )
    Group3_4 = Group(
        Item(name="img_4_p1"),
        Item(name="img_4_p2"),
        Item(name="img_4_p3"),
        Item(name="img_4_p4"),
        orientation="horizontal",
        label="Image 4",
        show_border=True,
    )
    Group3 = Group(
        Group3_1,
        Group3_2,
        Group3_3,
        Group3_4,
        show_border=True,
        label="Manual pre-orientation",
    )

    # calibration orientation param.

    Examine_Flag = Bool(False, label="Calibrate with different Z")
    Combine_Flag = Bool(False, label="Combine preprocessed planes")

    point_number_of_orientation = Int(label="Point number of orientation")
    cc = Bool(False, label="cc")
    xh = Bool(False, label="xh")
    yh = Bool(False, label="yh")
    k1 = Bool(False, label="k1")
    k2 = Bool(False, label="k2")
    k3 = Bool(False, label="k3")
    p1 = Bool(False, label="p1")
    p2 = Bool(False, label="p2")
    scale = Bool(False, label="scale")
    shear = Bool(False, label="shear")
    interf = Bool(False, label="interf (fit glass-interface tilt)")

    Group4_0 = Group(
        Item(name="Examine_Flag"), Item(name="Combine_Flag"), show_border=True
    )

    Group4_1 = Group(
        Item(name="cc"),
        Item(name="xh"),
        Item(name="yh"),
        orientation="vertical",
        columns=3,
    )
    Group4_2 = Group(
        Item(name="k1"),
        Item(name="k2"),
        Item(name="k3"),
        Item(name="p1"),
        Item(name="p2"),
        orientation="vertical",
        columns=5,
        label="Lens distortion(Brown)",
        show_border=True,
    )
    Group4_3 = Group(
        Item(name="scale"),
        Item(name="shear"),
        orientation="vertical",
        columns=2,
        label="Affin transformation",
        show_border=True,
    )
    Group4_4 = Group(
        Item(name="interf"),
        label="Glass-interface tilt",
        show_border=True,
    )

    Group4 = Group(
        Group(
            Group4_0,
            Item(name="point_number_of_orientation"),
            Group4_1,
            Group4_2,
            Group4_3,
            Group4_4,
            label=" Orientation Parameters ",
            show_border=True,
        ),
        orientation="horizontal",
        show_border=True,
        label="Calibration Orientation Param.",
    )

    dumbbell_eps = Float(label="dumbbell epsilon")
    dumbbell_scale = Float(label="dumbbell scale")
    dumbbell_gradient_descent = Float(label="dumbbell gradient descent factor")
    dumbbell_penalty_weight = Float(label="weight for dumbbell penalty")
    dumbbell_step = Int(label="step size through sequence")
    dumbbell_niter = Int(label="number of iterations per click")
    dumbbell_fixed_camera = Int(label="fixed camera (0=auto)")

    Group5 = HGroup(
        VGroup(
            Item(name="dumbbell_eps"),
            Item(name="dumbbell_scale"),
            Item(name="dumbbell_gradient_descent"),
            Item(name="dumbbell_penalty_weight"),
            Item(name="dumbbell_step"),
            Item(name="dumbbell_niter"),
            Item(name="dumbbell_fixed_camera"),
        ),
        spring,
        label="Dumbbell calibration parameters",
        show_border=True,
    )

    shaking_first_frame = Int(label="shaking first frame")
    shaking_last_frame = Int(label="shaking last frame")
    shaking_max_num_points = Int(label="shaking max num points")
    shaking_max_num_frames = Int(label="shaking max num frames")
    shaking_tol_px = Float(2.0, label="match tolerance (px)")
    shaking_hold_cam = Int(1, label="held camera (1..N, fixes gauge)")

    Group6 = HGroup(
        VGroup(
            Item(
                name="shaking_first_frame",
            ),
            Item(name="shaking_last_frame"),
            Item(name="shaking_max_num_points"),
            Item(name="shaking_max_num_frames"),
            Item(name="shaking_tol_px"),
            Item(name="shaking_hold_cam"),
        ),
        spring,
        label="Shaking calibration parameters",
        show_border=True,
    )

    Calib_Params_View = View(
        Tabbed(Group1, Group2, Group3, Group4, Group5, Group6),
        buttons=["Undo", "OK", "Cancel"],
        handler=CalHandler(),
        title="Calibration Parameters",
    )

    def _reload(self, num_cams, pm):
        global_n_cam = num_cams

        ptv_params = pm.get_section("ptv")
        self.h_image_size = int(ptv_params["imx"])
        self.v_image_size = int(ptv_params["imy"])
        self.h_pixel_size = ptv_params["pix_x"]
        self.v_pixel_size = ptv_params["pix_y"]
        self.hp_flag = bool(ptv_params["hp_flag"])

        cal_ori_params = pm.get_section("cal_ori")
        cal_names = list(cal_ori_params.get("img_cal_name", []))
        cal_names += [""] * max(0, global_n_cam - len(cal_names))
        for i in range(global_n_cam):
            setattr(self, f"cam_{i + 1}", cal_names[i])

        ori_names = list(cal_ori_params.get("img_ori", []))
        ori_names += [""] * max(0, global_n_cam - len(ori_names))
        for i in range(global_n_cam):
            setattr(self, f"ori_cam_{i + 1}", ori_names[i])

        self.fixp_name = cal_ori_params["fixp_name"]
        self._cal_splitter = bool(cal_ori_params["cal_splitter"])

        detect_plate_params = pm.get_section("detect_plate")
        self.grey_value_treshold_1 = int(detect_plate_params["gvth_1"])
        self.grey_value_treshold_2 = int(detect_plate_params["gvth_2"])
        self.grey_value_treshold_3 = int(detect_plate_params["gvth_3"])
        self.grey_value_treshold_4 = int(detect_plate_params["gvth_4"])
        self.tolerable_discontinuity = int(detect_plate_params["tol_dis"])
        self.min_npix = int(detect_plate_params["min_npix"])
        self.max_npix = int(detect_plate_params["max_npix"])
        self.min_npix_x = int(detect_plate_params["min_npix_x"])
        self.max_npix_x = int(detect_plate_params["max_npix_x"])
        self.min_npix_y = int(detect_plate_params["min_npix_y"])
        self.max_npix_y = int(detect_plate_params["max_npix_y"])
        self.sum_of_grey = int(detect_plate_params["sum_grey"])
        self.size_of_crosses = int(detect_plate_params["size_cross"])

        man_ori_params = pm.get_section("man_ori")
        nr = man_ori_params["nr"]
        for i in range(global_n_cam):
            for j in range(4):
                val = nr[i * 4 + j]
                setattr(self, f"img_{i + 1}_p{j + 1}", int(val))

        examine_params = pm.get_section("examine")
        self.Examine_Flag = bool(examine_params["Examine_Flag"])
        self.Combine_Flag = bool(examine_params["Combine_Flag"])

        orient_params = pm.get_section("orient")
        self.point_number_of_orientation = int(orient_params["pnfo"])
        self.cc = bool(orient_params["cc"])
        self.xh = bool(orient_params["xh"])
        self.yh = bool(orient_params["yh"])
        self.k1 = bool(orient_params["k1"])
        self.k2 = bool(orient_params["k2"])
        self.k3 = bool(orient_params["k3"])
        self.p1 = bool(orient_params["p1"])
        self.p2 = bool(orient_params["p2"])
        self.scale = bool(orient_params["scale"])
        self.shear = bool(orient_params["shear"])
        self.interf = bool(orient_params["interf"])

        dumbbell_params = pm.get_section("dumbbell")
        self.dumbbell_eps = dumbbell_params["dumbbell_eps"]
        self.dumbbell_scale = dumbbell_params["dumbbell_scale"]
        self.dumbbell_gradient_descent = dumbbell_params["dumbbell_gradient_descent"]
        self.dumbbell_penalty_weight = dumbbell_params["dumbbell_penalty_weight"]
        self.dumbbell_step = int(dumbbell_params["dumbbell_step"])
        self.dumbbell_niter = int(dumbbell_params["dumbbell_niter"])
        self.dumbbell_fixed_camera = int(
            dumbbell_params.get("dumbbell_fixed_camera", 0)
        )

        shaking_params = pm.get_section("shaking")
        self.shaking_first_frame = int(shaking_params["shaking_first_frame"])
        self.shaking_last_frame = int(shaking_params["shaking_last_frame"])
        self.shaking_max_num_points = int(shaking_params["shaking_max_num_points"])
        self.shaking_max_num_frames = int(shaking_params["shaking_max_num_frames"])
        self.shaking_tol_px = float(shaking_params.get("shaking_tol_px", 2.0))
        self.shaking_hold_cam = int(shaking_params.get("shaking_hold_cam", 1))

    def __init__(self, experiment: Experiment):
        HasTraits.__init__(self)
        self.experiment = experiment
        self._reload(experiment.get_n_cam(), experiment.pm)


# Experiment and Paramset classes moved to experiment.py for better separation of concerns
