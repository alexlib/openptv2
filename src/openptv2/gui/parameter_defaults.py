"""Default parameter values for OpenPTV.

This is the SINGLE PLACE to edit default values.
All converters import from here.

Key principle:
- None = required (no default - must be in YAML)
- value = optional with default
"""

# ============ REQUIRED PARAMETERS (no default - must be in YAML) ============

DEFAULT_PTV = {
    "imx": None,
    "imy": None,
    "pix_x": None,
    "pix_y": None,
}

DEFAULT_SEQUENCE = {
    "first": None,
    "last": None,
    "base_name": None,
}

DEFAULT_CRITERIA = {
    "X_lay": None,
}

DEFAULT_CAL_ORI = {
    "img_cal_name": None,
    "img_ori": None,
}

# ============ OPTIONAL PARAMETERS (have defaults) ============

DEFAULT_PTV_OPTIONAL = {
    "hp_flag": True,
    "highpass_size": 25,  # low-pass box dim subtracted in simple_highpass
    "tiff_flag": True,
    "chfield": 0,
    "allcam_flag": False,
    "splitter": False,
    "splitter_order": [0, 1, 3, 2],
    # Multimedia parameters
    "mmp_n1": 1.0,
    "mmp_n2": 1.33,
    "mmp_n3": 1.46,
    "mmp_d": 6.0,
}

DEFAULT_CRITERIA_OPTIONAL = {
    "Zmin_lay": [-50],
    "Zmax_lay": [50],
    "cn": 0.0,
    "cnx": 0.0,
    "cny": 0.0,
    "csumg": 0.0,
    "eps0": 0.0,
    "corrmin": 0.0,
}

DEFAULT_TRACK = {
    "preset": "full_multipass",
    "dvxmin": -20,
    "dvxmax": 20,
    "dvymin": -20,
    "dvymax": 20,
    "dvzmin": -20,
    "dvzmax": 20,
    "dangle": 10,
    "dacc": 2,
    "add": 0,
    "dsumg": 0,
    "dn": 1,
    "dnx": 0,
    "dny": 0,
    "angle": 10,
    "flagNewParticles": True,
    "track_mode": 0,
    "postprocess": True,
}

DEFAULT_DETECT_PLATE = {
    "gvth_1": 40,
    "gvth_2": 40,
    "gvth_3": 40,
    "gvth_4": 40,
    "min_npix": 25,
    "max_npix": 400,
    "size_cross": 3,
    "sum_grey": 100,
    "min_npix_x": 5,
    "max_npix_x": 50,
    "min_npix_y": 5,
    "max_npix_y": 50,
}

DEFAULT_TARG_REC = {
    "discont": 100,
    "nnmin": 4,
    "nnmax": 500,
    "nxmin": 2,
    "nxmax": 100,
    "nymin": 2,
    "nymax": 100,
    "sumg_min": 150,
    "cr_sz": 2,
}

DEFAULT_CAL_ORI_OPTIONAL = {
    "fixp_name": "",
    "tiff_flag": True,
    "pair_flag": False,
    "chfield": 0,
    "cal_splitter": False,
}

DEFAULT_ORIENT = {
    "useflag": 0,
    "ccflag": 0,
    "xhflag": 0,
    "yhflag": 0,
    "k1flag": 0,
    "k2flag": 0,
    "k3flag": 0,
    "p1flag": 0,
    "p2flag": 0,
    "scxflag": 0,
    "sheflag": 0,
    "interfflag": 0,
}

DEFAULT_EXAMINE = {
    "Examine_Flag": False,
    "Combine_Flag": False,
}

DEFAULT_PFT_VERSION = {
    "Existing_Target": 0,
}

DEFAULT_MULTI_PLANES = {
    "n_planes": 0,
    "plane_name": [],
}

DEFAULT_MASKING = {
    "mask_flag": False,
    "mask_base_name": "",
}

DEFAULT_UNSHARP_MASK = {
    "flag": False,
    "size": 3,
    "strength": 1.0,
}

DEFAULT_PLUGINS = {
    "available_tracking": ["default"],
    "available_sequence": ["default"],
    "selected_tracking": "default",
    "selected_sequence": "default",
}

DEFAULT_DUMBELL = {
    "dumbbell_eps": 3.0,
    "dumbbell_gradient_descent": 0.05,
    "dumbbell_niter": 500,
    "dumbbell_penalty_weight": 1.0,
    "dumbbell_scale": 25.0,
    "dumbbell_step": 1,
}

DEFAULT_SHAKING = {
    "shaking_first_frame": 0,
    "shaking_last_frame": 0,
    "shaking_max_num_frames": 5,
    "shaking_max_num_points": 10,
}

DEFAULT_SORTGRID = {
    "radius": 20,
}
