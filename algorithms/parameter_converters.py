"""Parameter converters from YAML dict to algorithm dataclasses.

This module is the SINGLE PLACE for converting YAML parameters (with key variations)
to Python algorithm dataclasses.

Design principles:
1. Required parameters MUST be in YAML -> ValueError if missing
2. Optional parameters use defaults from parameter_defaults.py
3. All key variations handled here (criteria/volume, X_lay/x_lay)
4. Backward compatible with existing YAML files

Usage:
    from algorithms.parameter_converters import (
        get_control_par, get_sequence_par, get_volume_par, get_track_par_tuple,
    )

    pm = experiment.pm
    params = pm.parameters

    cpar = get_control_par(params)
    vpar = get_volume_par(params)
    tpar = get_track_par_tuple(params)
    spar = get_sequence_par(params)
"""

import numpy as np

from gui.pyptv.parameter_defaults import (
    DEFAULT_PTV,
    DEFAULT_PTV_OPTIONAL,
    DEFAULT_SEQUENCE,
    DEFAULT_CRITERIA,
    DEFAULT_CRITERIA_OPTIONAL,
    DEFAULT_TRACK,
    DEFAULT_DETECT_PLATE,
    DEFAULT_TARG_REC,
    DEFAULT_CAL_ORI,
    DEFAULT_CAL_ORI_OPTIONAL,
    DEFAULT_ORIENT,
    DEFAULT_EXAMINE,
    DEFAULT_PFT_VERSION,
    DEFAULT_MULTI_PLANES,
    DEFAULT_MASKING,
    DEFAULT_UNSHARP_MASK,
    DEFAULT_PLUGINS,
    DEFAULT_DUMBELL,
    DEFAULT_SHAKING,
    DEFAULT_SORTGRID,
)

from .parameters import (
    ControlPar,
    SequencePar,
    VolumePar,
    TrackParTuple,
    TargetPar,
    CalibrationPar,
    OrientPar,
    MultiPlanesPar,
    ExaminePar,
    PftVersionPar,
    MultimediaPar,
)


def convert_optv_calibrations(cals):
    """Convert optv.Calibration objects to Python Calibration objects.

    This is needed when passing calibrations from the GUI (which uses optv.Calibration)
    to the Python Tracker (which uses algorithms.calibration.Calibration).

    Arguments:
        cals: List of optv.calibration.Calibration objects

    Returns:
        List of algorithms.calibration.Calibration objects
    """
    from algorithms.calibration import (
        Calibration as PythonCalibration,
        Exterior,
        Interior,
    )

    py_cals = []
    for cal in cals:
        try:
            ext_par = PythonCalibration._create_default_exterior()

            pos = cal.get_pos()
            angles = cal.get_angles()

            ext_par["x0"][()] = pos[0]
            ext_par["y0"][()] = pos[1]
            ext_par["z0"][()] = pos[2]
            ext_par["omega"][()] = angles[0]
            ext_par["phi"][()] = angles[1]
            ext_par["kappa"][()] = angles[2]

            from algorithms.calibration import rotation_matrix

            rotation_matrix(ext_par)

            int_par = PythonCalibration._create_default_interior()
            prim_point = cal.get_primary_point()
            int_par.xh[()] = prim_point[0]
            int_par.yh[()] = prim_point[1]
            int_par.cc[()] = prim_point[2]

            glass_par = cal.get_glass_vec()

            rad_dist = cal.get_radial_distortion()
            decent = cal.get_decentering()
            affine = cal.get_affine()
            added_par = np.array(
                list(rad_dist) + list(decent) + list(affine), dtype=np.float64
            )

            py_cal = PythonCalibration(
                ext_par=ext_par,
                int_par=int_par,
                glass_par=glass_par,
                added_par=added_par,
            )
            py_cals.append(py_cal)
            print(
                f"[DEBUG] Converted cal {len(py_cals)}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            )

        except Exception as e:
            import traceback

            print(f"[DEBUG] Error converting calibration: {e}")
            traceback.print_exc()
            py_cals.append(PythonCalibration())

    return py_cals


from .parameters import (
    ControlPar,
    SequencePar,
    VolumePar,
    TrackParTuple,
    TargetPar,
    CalibrationPar,
    OrientPar,
    MultiPlanesPar,
    ExaminePar,
    PftVersionPar,
    MultimediaPar,
)


def _get_section(yaml_params, *keys):
    """Get a section from YAML params, trying multiple key names.

    Example: _get_section(params, 'criteria', 'volume')
    """
    for key in keys:
        if key in yaml_params:
            return yaml_params[key] or {}
    return {}


def _check_required(file_dict, required_defaults, section_name):
    """Check for required keys that are None in defaults.

    Returns list of missing required keys.
    Case-insensitive key lookup is used.
    """
    lower_keys = {k.lower(): v for k, v in file_dict.items()}
    missing = []
    for key, default_value in required_defaults.items():
        if default_value is None:
            value = file_dict.get(key) or lower_keys.get(key.lower())
            if value is None or (isinstance(value, list) and len(value) == 0):
                missing.append(key)
    return missing


def _merge_with_defaults(file_dict, optional_defaults):
    """Merge file values with optional defaults.

    File values override defaults, but None values in file are skipped.
    """
    result = {**optional_defaults}
    result.update({k: v for k, v in file_dict.items() if v is not None})
    return result


# ============ Converters ============


def get_multimedia_par(yaml_params):
    """Get MultimediaPar from YAML parameters."""
    ptv = yaml_params.get("ptv") or {}
    opt = _merge_with_defaults(ptv, DEFAULT_PTV_OPTIONAL)

    return MultimediaPar(
        n1=opt.get("mmp_n1", 1.0),
        n2=[opt.get("mmp_n2", 1.33)],
        d=[opt.get("mmp_d", 6.0)],
        n3=opt.get("mmp_n3", 1.46),
    )


def get_control_par(yaml_params):
    """Get ControlPar from YAML parameters.

    Required in ptv section: imx, imy, pix_x, pix_y

    Raises ValueError if required params missing.
    """
    ptv = yaml_params.get("ptv") or {}
    num_cams = yaml_params.get("num_cams", 4)

    missing = _check_required(ptv, DEFAULT_PTV, "ptv")
    if missing:
        raise ValueError(f"Missing required ptv parameters: {missing}")

    opt = _merge_with_defaults(ptv, DEFAULT_PTV_OPTIONAL)
    mm = get_multimedia_par(yaml_params)

    cpar = ControlPar(
        num_cams=num_cams,
        imx=ptv["imx"],
        imy=ptv["imy"],
        pix_x=ptv["pix_x"],
        pix_y=ptv["pix_y"],
        hp_flag=1 if opt.get("hp_flag", True) else 0,
        all_cam_flag=1 if opt.get("allcam_flag", False) else 0,
        tiff_flag=1 if opt.get("tiff_flag", True) else 0,
        chfield=opt.get("chfield", 0),
    )
    cpar.mm = mm
    return cpar


def get_sequence_par(yaml_params):
    """Get SequencePar from YAML parameters.

    Required: first, last, base_name

    Raises ValueError if required params missing.
    """
    seq = yaml_params.get("sequence") or {}

    missing = _check_required(seq, DEFAULT_SEQUENCE, "sequence")
    if missing:
        raise ValueError(f"Missing required sequence parameters: {missing}")

    return SequencePar(
        first=seq["first"],
        last=seq["last"],
        img_base_name=seq["base_name"],
    )


def get_volume_par(yaml_params):
    """Get VolumePar from YAML parameters.

    Key variations: criteria/volume, X_lay/x_lay, Zmin_lay/z_min_lay

    Required: X_lay (or x_lay)

    Raises ValueError if required params missing.
    """
    vol = _get_section(yaml_params, "criteria", "volume")

    missing = _check_required(vol, DEFAULT_CRITERIA, "criteria")
    if missing:
        raise ValueError(f"Missing required criteria parameters: {missing}")

    opt = _merge_with_defaults(vol, DEFAULT_CRITERIA_OPTIONAL)

    z_min = vol.get("Zmin_lay") or vol.get("z_min_lay") or opt.get("Zmin_lay", [-50])
    z_max = vol.get("Zmax_lay") or vol.get("z_max_lay") or opt.get("Zmax_lay", [50])

    x_lay = vol.get("X_lay") or vol.get("x_lay")

    return VolumePar(
        X_lay=x_lay,
        Zmin_lay=z_min,
        Zmax_lay=z_max,
        cn=opt.get("cn", 0),
        cnx=opt.get("cnx", 0),
        cny=opt.get("cny", 0),
        csumg=opt.get("csumg", 0),
        eps0=opt.get("eps0", 0),
        corrmin=opt.get("corrmin", 0),
    )


def get_track_par_tuple(yaml_params):
    """Get TrackParTuple from YAML parameters.

    Key variations: track/tracking, dangle/angle

    All parameters have defaults - none required.
    """
    track = _get_section(yaml_params, "track", "tracking")
    t = _merge_with_defaults(track, DEFAULT_TRACK)

    # Prefer user-supplied "angle" over default "dangle"
    if "angle" in track:
        dangle = track["angle"]
    elif "dangle" in track:
        dangle = track["dangle"]
    else:
        dangle = t.get("dangle", 10)

    return TrackParTuple(
        dvxmin=t["dvxmin"],
        dvxmax=t["dvxmax"],
        dvymin=t["dvymin"],
        dvymax=t["dvymax"],
        dvzmin=t["dvzmin"],
        dvzmax=t["dvzmax"],
        dangle=dangle,
        dacc=t["dacc"],
        add=t["add"],
        dsumg=t["dsumg"],
        dn=t["dn"],
        dnx=t["dnx"],
        dny=t["dny"],
    )


def get_target_par(yaml_params):
    """Get TargetPar from YAML parameters."""
    targ = yaml_params.get("targ_rec") or yaml_params.get("targ", {})
    plate = yaml_params.get("detect_plate") or yaml_params.get("plate", {})

    t = _merge_with_defaults(targ, DEFAULT_TARG_REC)
    p = _merge_with_defaults(plate, DEFAULT_DETECT_PLATE)

    return TargetPar(
        gvthres=[
            p.get("gvth_1", 40),
            p.get("gvth_2", 40),
            p.get("gvth_3", 40),
            p.get("gvth_4", 40),
        ],
        discont=t["discont"],
        nnmin=t["nnmin"],
        nnmax=t["nnmax"],
        nxmin=t["nxmin"],
        nxmax=t["nxmax"],
        nymin=t["nymin"],
        nymax=t["nymax"],
        sumg_min=t["sumg_min"],
        cr_sz=t["cr_sz"],
    )


def get_calibration_par(yaml_params):
    """Get CalibrationPar from YAML parameters.

    Required: img_cal_name, img_ori

    Raises ValueError if required params missing.
    """
    cal = _get_section(yaml_params, "cal_ori", "calib")

    missing = _check_required(cal, DEFAULT_CAL_ORI, "cal_ori")
    if missing:
        raise ValueError(f"Missing required cal_ori parameters: {missing}")

    opt = _merge_with_defaults(cal, DEFAULT_CAL_ORI_OPTIONAL)

    return CalibrationPar(
        fixp_name=opt.get("fixp_name", ""),
        img_name=cal["img_cal_name"],
        img_ori0=cal["img_ori"],
        tiff_flag=1 if opt.get("tiff_flag", True) else 0,
        pair_flag=1 if opt.get("pair_flag", False) else 0,
        chfield=opt.get("chfield", 0),
    )


def get_orient_par(yaml_params):
    """Get OrientPar from YAML parameters."""
    orient = yaml_params.get("orient") or {}
    o = _merge_with_defaults(orient, DEFAULT_ORIENT)

    return OrientPar(
        useflag=o["useflag"],
        ccflag=o["ccflag"],
        xhflag=o["xhflag"],
        yhflag=o["yhflag"],
        k1flag=o["k1flag"],
        k2flag=o["k2flag"],
        k3flag=o["k3flag"],
        p1flag=o["p1flag"],
        p2flag=o["p2flag"],
        scxflag=o["scxflag"],
        sheflag=o["sheflag"],
        interfflag=o["interfflag"],
    )


def get_multiplanes_par(yaml_params):
    """Get MultiPlanesPar from YAML parameters."""
    mp = yaml_params.get("multi_planes") or yaml_params.get("multi_planes", {})
    m = _merge_with_defaults(mp, DEFAULT_MULTI_PLANES)

    return MultiPlanesPar(
        num_planes=m["n_planes"],
        filename=m["plane_name"],
    )


def get_examine_par(yaml_params):
    """Get ExaminePar from YAML parameters."""
    exam = yaml_params.get("examine") or {}
    e = _merge_with_defaults(exam, DEFAULT_EXAMINE)

    return ExaminePar(
        examine_flag=e["Examine_Flag"],
        combine_flag=e["Combine_Flag"],
    )


def get_pft_version_par(yaml_params):
    """Get PftVersionPar from YAML parameters."""
    pft = yaml_params.get("pft_version") or {}
    p = _merge_with_defaults(pft, DEFAULT_PFT_VERSION)

    return PftVersionPar(
        existing_target_flag=p["Existing_Target"],
    )


def get_all_params(yaml_params):
    """Get all algorithm parameters at once.

    Returns a dict with all parameter objects.
    Useful for debugging or batch operations.
    """
    return {
        "cpar": get_control_par(yaml_params),
        "spar": get_sequence_par(yaml_params),
        "vpar": get_volume_par(yaml_params),
        "tpar": get_track_par_tuple(yaml_params),
        "targpar": get_target_par(yaml_params),
        "calpar": get_calibration_par(yaml_params),
        "orientpar": get_orient_par(yaml_params),
        "multiplanespar": get_multiplanes_par(yaml_params),
        "examinepar": get_examine_par(yaml_params),
        "pftversionpar": get_pft_version_par(yaml_params),
    }
