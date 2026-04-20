"""Tracking algorithms — Python translation of lib/src/track.c."""

import numpy as np

from .constants import (
    MAX_CANDS, PT_UNUSED, TR_UNUSED, CORRES_NONE, PREV_NONE, NEXT_NONE,
    COORD_UNUSED, TR_BUFSPACE, TR_MAX_CAMS, ADD_PART,
)
from .tracking_frame_buf import register_link_candidate, reset_links

Foundpix_dtype = np.dtype([
    ("ftnr", np.int32),
    ("freq", np.int32),
    ("whichcam", np.int32, (4,)),
])


def predict(prev_pos, curr_pos, c):
    prev_pos = np.asarray(prev_pos)
    curr_pos = np.asarray(curr_pos)
    c[:] = curr_pos + (curr_pos - prev_pos)


def search_volume_center_moving(prev_pos, curr_pos):
    prev_pos = np.asarray(prev_pos)
    curr_pos = np.asarray(curr_pos)
    return curr_pos + (curr_pos - prev_pos)


def pos3d_in_bounds(pos, bounds):
    x, y, z = pos
    return bool(
        bounds.dvxmin < x < bounds.dvxmax and
        bounds.dvymin < y < bounds.dvymax and
        bounds.dvzmin < z < bounds.dvzmax
    )


def angle_acc(start, pred, cand):
    v0 = np.asarray(pred) - np.asarray(start)
    v1 = np.asarray(cand) - np.asarray(start)

    if np.array_equal(v0, -v1):
        angle = 200.0
    elif np.array_equal(v0, v1):
        angle = 0.0
    else:
        norm0 = np.linalg.norm(v0)
        norm1 = np.linalg.norm(v1)
        if norm0 == 0 or norm1 == 0:
            angle = 0.0
        else:
            dot = np.dot(v0, v1) / (norm0 * norm1)
            dot = np.clip(dot, -1.0, 1.0)
            angle = np.arccos(dot) * 200.0 / np.pi

    acc = np.linalg.norm(v1 - v0)
    return angle, acc


def candsearch_in_pix(next_targets, num_targets, cent_x, cent_y,
                      dl, dr, du, dd, cpar):
    p = [PT_UNUSED] * 4

    xmin = cent_x - dl
    xmax = cent_x + dr
    ymin = cent_y - du
    ymax = cent_y + dd

    if xmin < 0.0:
        xmin = 0.0
    if xmax > cpar.imx:
        xmax = cpar.imx
    if ymin < 0.0:
        ymin = 0.0
    if ymax > cpar.imy:
        ymax = cpar.imy

    p1 = p2 = p3 = p4 = PT_UNUSED
    dmin = 1e20
    d1 = d2 = d3 = d4 = dmin

    if not (0.0 <= cent_x <= cpar.imx and 0.0 <= cent_y <= cpar.imy):
        return p

    j0 = num_targets // 2
    dj = num_targets // 4
    while dj > 1:
        if next_targets[j0].y < ymin:
            j0 += dj
        else:
            j0 -= dj
        dj //= 2

    j0 -= 12
    if j0 < 0:
        j0 = 0

    for j in range(j0, num_targets):
        t = next_targets[j]
        if t.tnr != TR_UNUSED:
            if t.y > ymax:
                break
            if t.x > xmin and t.x < xmax and t.y > ymin and t.y < ymax:
                d = np.sqrt((cent_x - t.x) ** 2 + (cent_y - t.y) ** 2)

                if d < dmin:
                    dmin = d

                if d < d1:
                    p4, p3, p2, p1 = p3, p2, p1, j
                    d4, d3, d2, d1 = d3, d2, d1, d
                elif d1 < d < d2:
                    p4, p3, p2 = p3, p2, j
                    d4, d3, d2 = d3, d2, d
                elif d2 < d < d3:
                    p4, p3 = p3, j
                    d4, d3 = d3, d
                elif d3 < d < d4:
                    p4 = j
                    d4 = d

    p[0], p[1], p[2], p[3] = p1, p2, p3, p4
    return p


def candsearch_in_pix_rest(next_targets, num_targets, cent_x, cent_y,
                           dl, dr, du, dd, p, cpar):
    xmin = cent_x - dl
    xmax = cent_x + dr
    ymin = cent_y - du
    ymax = cent_y + dd

    if xmin < 0.0:
        xmin = 0.0
    if xmax > cpar.imx:
        xmax = cpar.imx
    if ymin < 0.0:
        ymin = 0.0
    if ymax > cpar.imy:
        ymax = cpar.imy

    p[0] = PT_UNUSED
    counter = 0
    dmin = 1e20

    if not (0.0 <= cent_x <= cpar.imx and 0.0 <= cent_y <= cpar.imy):
        return 0

    j0 = num_targets // 2
    dj = num_targets // 4
    while dj > 1:
        if next_targets[j0].y < ymin:
            j0 += dj
        else:
            j0 -= dj
        dj //= 2

    j0 -= 12
    if j0 < 0:
        j0 = 0

    for j in range(j0, num_targets):
        t = next_targets[j]
        if t.tnr == TR_UNUSED:
            if t.y > ymax:
                break
            if t.x > xmin and t.x < xmax and t.y > ymin and t.y < ymax:
                d = np.sqrt((cent_x - t.x) ** 2 + (cent_y - t.y) ** 2)
                if d < dmin:
                    dmin = d
                    p[0] = j
                    counter = 1

    return counter


def reset_foundpix_array(arr, n, num_cams):
    for i in range(n):
        arr[i]['ftnr'] = TR_UNUSED
        arr[i]['freq'] = 0
        arr[i]['whichcam'][:num_cams] = [0] * num_cams


def copy_foundpix_array(dest, src, n, num_cams):
    for i in range(n):
        dest[i]['ftnr'] = src[i]['ftnr']
        dest[i]['freq'] = src[i]['freq']
        dest[i]['whichcam'][:num_cams] = src[i]['whichcam'][:num_cams]


def sort_candidates_by_freq(items, num_cams):
    n = num_cams * MAX_CANDS

    for i in range(n):
        for j in range(num_cams):
            for m in range(MAX_CANDS):
                if items[i]['ftnr'] == items[4 * j + m]['ftnr']:
                    items[i]['whichcam'][j] = 1

    for i in range(n):
        for j in range(num_cams):
            if items[i]['whichcam'][j] == 1 and items[i]['ftnr'] != TR_UNUSED:
                items[i]['freq'] += 1

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            if items[j - 1]['freq'] < items[j]['freq']:
                items[j - 1], items[j] = items[j].copy(), items[j - 1].copy()

    for i in range(n):
        for j in range(i + 1, n):
            if items[i]['ftnr'] == items[j]['ftnr'] or items[j]['freq'] < 2:
                items[j]['freq'] = 0
                items[j]['ftnr'] = TR_UNUSED

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            if items[j - 1]['freq'] < items[j]['freq']:
                items[j - 1], items[j] = items[j].copy(), items[j - 1].copy()

    different = 0
    for i in range(n):
        if items[i]['freq'] != 0:
            different += 1
    return different


def sort(n, a, b):
    """Bubble sort arrays a and b by ascending a values, in-place (matches C)."""
    flag = True
    while flag:
        flag = False
        for i in range(n - 1):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                b[i], b[i + 1] = b[i + 1], b[i]
                flag = True


def point_to_pixel(point, cal, cpar):
    from .imgcoord import img_coord
    from .trafo import metric_to_pixel
    x, y = img_coord(point, cal, cpar.mm)
    return metric_to_pixel(x, y, cpar)


def searchquader(point, tpar, cpar, calib):
    num_cams = cpar.num_cams
    xr = np.zeros(num_cams)
    xl = np.zeros(num_cams)
    yd = np.zeros(num_cams)
    yu = np.zeros(num_cams)

    mins = np.array([tpar.dvxmin, tpar.dvymin, tpar.dvzmin])
    maxes = np.array([tpar.dvxmax, tpar.dvymax, tpar.dvzmax])

    quader = np.zeros((8, 3))
    for pt in range(8):
        quader[pt] = point.copy()
        for dim in range(3):
            if pt & (1 << dim):
                quader[pt, dim] += maxes[dim]
            else:
                quader[pt, dim] += mins[dim]

    for i in range(num_cams):
        xr[i] = 0
        xl[i] = cpar.imx
        yd[i] = 0
        yu[i] = cpar.imy

        cx, cy = point_to_pixel(point, calib[i], cpar)

        for pt in range(8):
            corner_x, corner_y = point_to_pixel(quader[pt], calib[i], cpar)

            if corner_x < xl[i]:
                xl[i] = corner_x
            if corner_y < yu[i]:
                yu[i] = corner_y
            if corner_x > xr[i]:
                xr[i] = corner_x
            if corner_y > yd[i]:
                yd[i] = corner_y

        if xl[i] < 0:
            xl[i] = 0
        if yu[i] < 0:
            yu[i] = 0
        if xr[i] > cpar.imx:
            xr[i] = cpar.imx
        if yd[i] > cpar.imy:
            yd[i] = cpar.imy

        xr[i] = xr[i] - cx
        xl[i] = cx - xl[i]
        yd[i] = yd[i] - cy
        yu[i] = cy - yu[i]

    return xr, xl, yd, yu


def register_closest_neighbs(targets, num_targets, cam, cent_x, cent_y,
                             dl, dr, du, dd, reg, cpar):
    all_cands = candsearch_in_pix(targets, num_targets, cent_x, cent_y,
                                  dl, dr, du, dd, cpar)
    for cand in range(MAX_CANDS):
        if all_cands[cand] == PT_UNUSED:
            reg[cand]['ftnr'] = TR_UNUSED
        else:
            reg[cand]['whichcam'][cam] = 1
            reg[cand]['ftnr'] = targets[all_cands[cand]].tnr


def sorted_candidates_in_volume(center, center_proj, frm, run):
    num_cams = frm.num_cams
    points = np.zeros(num_cams * MAX_CANDS, dtype=Foundpix_dtype).view(np.recarray)
    reset_foundpix_array(points, num_cams * MAX_CANDS, num_cams)

    xr, xl, yd, yu = searchquader(center, run.tpar, run.cpar, run.cal)

    for cam in range(num_cams):
        register_closest_neighbs(
            frm.targets[cam], frm.num_targets[cam], cam,
            center_proj[cam][0], center_proj[cam][1],
            xl[cam], xr[cam], yu[cam], yd[cam],
            points[cam * MAX_CANDS:(cam + 1) * MAX_CANDS], run.cpar)

    num_cands = sort_candidates_by_freq(points, num_cams)
    if num_cands > 0:
        result = np.zeros(num_cands + 1, dtype=Foundpix_dtype).view(np.recarray)
        for i in range(num_cands):
            result[i]['ftnr'] = points[i]['ftnr']
            result[i]['freq'] = points[i]['freq']
            result[i]['whichcam'][:] = points[i]['whichcam'][:]
        result[num_cands]['ftnr'] = TR_UNUSED
        return result
    return None


def assess_new_position(pos, targ_pos, cand_inds, frm, run):
    from .trafo import pixel_to_metric, dist_to_flat

    left = right = up = down = ADD_PART

    for cam in range(TR_MAX_CAMS):
        targ_pos[cam][0] = targ_pos[cam][1] = COORD_UNUSED

    for cam in range(run.cpar.num_cams):
        px, py = point_to_pixel(pos, run.cal[cam], run.cpar)

        num_cands = candsearch_in_pix_rest(
            frm.targets[cam], frm.num_targets[cam],
            px, py, left, right, up, down,
            cand_inds[cam], run.cpar)

        if num_cands > 0:
            _ix = cand_inds[cam][0]
            targ_pos[cam][0] = frm.targets[cam][_ix].x
            targ_pos[cam][1] = frm.targets[cam][_ix].y

    valid_cams = 0
    for cam in range(run.cpar.num_cams):
        if (targ_pos[cam][0] != COORD_UNUSED and
                targ_pos[cam][1] != COORD_UNUSED):
            mx, my = pixel_to_metric(targ_pos[cam][0], targ_pos[cam][1], run.cpar)
            cal = run.cal[cam]
            fx, fy = dist_to_flat(
                mx, my,
                cal.int_par.xh, cal.int_par.yh,
                cal.added_par.k1, cal.added_par.k2, cal.added_par.k3,
                cal.added_par.p1, cal.added_par.p2,
                cal.added_par.scx, cal.added_par.she,
                run.flatten_tol)
            targ_pos[cam][0] = fx
            targ_pos[cam][1] = fy
            valid_cams += 1
    return valid_cams


def add_particle(frm, pos, cand_inds):
    num_parts = frm.num_parts
    ref_path_inf = frm.path_info[num_parts]
    ref_path_inf.x[:] = pos
    reset_links(ref_path_inf)

    ref_corres = frm.correspond[num_parts]
    for cam in range(frm.num_cams):
        ref_corres.p[cam] = CORRES_NONE
        if cand_inds[cam][0] != PT_UNUSED:
            _ix = cand_inds[cam][0]
            frm.targets[cam][_ix].tnr = num_parts
            ref_corres.p[cam] = _ix
            ref_corres.nr = num_parts
    frm.num_parts += 1


def track_forward_start(run):
    for step in range(run.seq_par.first, run.seq_par.first + TR_BUFSPACE - 1):
        run.fb.read_frame_at_end(step, read_links=False)
        run.fb.fb_next()
    run.fb.fb_prev()


def trackcorr_c_loop(run_info, step):
    from .orientation import point_position

    fb = run_info.fb
    cal = run_info.cal
    tpar = run_info.tpar
    vpar = run_info.vpar
    cpar = run_info.cpar
    curr_targets = fb.buf[1].targets

    count1 = 0
    num_added = 0
    orig_parts = fb.buf[1].num_parts

    for h in range(orig_parts):
        X = [np.zeros(3) for _ in range(6)]

        curr_path_inf = fb.buf[1].path_info[h]
        curr_corres = fb.buf[1].correspond[h]
        curr_path_inf.inlist = 0

        X[1][:] = curr_path_inf.x

        v1 = [[0.0, 0.0] for _ in range(fb.num_cams)]

        if curr_path_inf.prev >= 0:
            ref_path_inf = fb.buf[0].path_info[curr_path_inf.prev]
            X[0][:] = ref_path_inf.x
            X[2][:] = search_volume_center_moving(ref_path_inf.x, curr_path_inf.x)

            for j in range(fb.num_cams):
                v1[j] = list(point_to_pixel(X[2], cal[j], cpar))
        else:
            X[2][:] = X[1]
            for j in range(fb.num_cams):
                if curr_corres.p[j] == CORRES_NONE:
                    v1[j] = list(point_to_pixel(X[2], cal[j], cpar))
                else:
                    _ix = curr_corres.p[j]
                    v1[j][0] = curr_targets[j][_ix].x
                    v1[j][1] = curr_targets[j][_ix].y

        w = sorted_candidates_in_volume(X[2], v1, fb.buf[2], run_info)
        if w is None:
            continue

        count2 = 0
        mm = 0
        while w[mm]['ftnr'] != TR_UNUSED:
            ref_path_inf = fb.buf[2].path_info[w[mm]['ftnr']]
            X[3][:] = ref_path_inf.x

            if curr_path_inf.prev >= 0:
                for j in range(3):
                    X[5][j] = 0.5 * (5.0 * X[3][j] - 4.0 * X[1][j] + X[0][j])
            else:
                X[5][:] = search_volume_center_moving(X[1], X[3])

            for j in range(fb.num_cams):
                v1[j] = list(point_to_pixel(X[5], cal[j], cpar))

            wn = sorted_candidates_in_volume(X[5], v1, fb.buf[3], run_info)
            if wn is not None:
                count3 = 0
                kk = 0
                while wn[kk]['ftnr'] != TR_UNUSED:
                    ref_path_inf = fb.buf[3].path_info[wn[kk]['ftnr']]
                    X[4][:] = ref_path_inf.x

                    diff_pos = X[4] - X[3]
                    if pos3d_in_bounds(diff_pos, tpar):
                        angle1, acc1 = angle_acc(X[3], X[4], X[5])
                        if curr_path_inf.prev >= 0:
                            angle0, acc0 = angle_acc(X[1], X[2], X[3])
                        else:
                            acc0 = acc1
                            angle0 = angle1

                        acc = (acc0 + acc1) / 2
                        angle = (angle0 + angle1) / 2
                        quali = wn[kk]['freq'] + w[mm]['freq']

                        if ((acc < tpar.dacc and angle < tpar.dangle) or
                                (acc < tpar.dacc / 10)):
                            dl = (np.linalg.norm(X[1] - X[3]) +
                                  np.linalg.norm(X[4] - X[3])) / 2
                            rr = (dl / run_info.lmax + acc / tpar.dacc +
                                  angle / tpar.dangle) / quali
                            register_link_candidate(curr_path_inf, rr, w[mm]['ftnr'])
                    kk += 1

            v2 = [[0.0, 0.0] for _ in range(TR_MAX_CAMS)]
            philf = [[PT_UNUSED] * MAX_CANDS for _ in range(TR_MAX_CAMS)]
            quali = assess_new_position(X[5], v2, philf, fb.buf[3], run_info)

            if quali >= 2:
                in_volume = 0
                v2_arr = np.array(v2[:cpar.num_cams], dtype=np.float64)
                X[4], dl = point_position(v2_arr, cpar.num_cams, cpar.mm, cal)

                if (vpar.X_lay[0] < X[4][0] < vpar.X_lay[1] and
                        run_info.ymin < X[4][1] < run_info.ymax and
                        vpar.Zmin_lay[0] < X[4][2] < vpar.Zmax_lay[1]):
                    in_volume = 1

                diff_pos = X[3] - X[4]
                if in_volume == 1 and pos3d_in_bounds(diff_pos, tpar):
                    angle, acc = angle_acc(X[3], X[4], X[5])
                    if ((acc < tpar.dacc and angle < tpar.dangle) or
                            (acc < tpar.dacc / 10)):
                        dl = (np.linalg.norm(X[1] - X[3]) +
                              np.linalg.norm(X[4] - X[3])) / 2
                        rr = (dl / run_info.lmax + acc / tpar.dacc +
                              angle / tpar.dangle) / (quali + w[mm]['freq'])
                        register_link_candidate(curr_path_inf, rr, w[mm]['ftnr'])

                        if tpar.add:
                            add_particle(fb.buf[3], X[4], philf)
                            num_added += 1
                in_volume = 0
            quali = 0

            if curr_path_inf.inlist == 0 and curr_path_inf.prev >= 0:
                diff_pos = X[3] - X[1]
                if pos3d_in_bounds(diff_pos, tpar):
                    angle, acc = angle_acc(X[1], X[2], X[3])
                    if ((acc < tpar.dacc and angle < tpar.dangle) or
                            (acc < tpar.dacc / 10)):
                        quali = w[mm]['freq']
                        dl = (np.linalg.norm(X[1] - X[3]) +
                              np.linalg.norm(X[0] - X[1])) / 2
                        rr = (dl / run_info.lmax + acc / tpar.dacc +
                              angle / tpar.dangle) / quali
                        register_link_candidate(curr_path_inf, rr, w[mm]['ftnr'])

            mm += 1

        if tpar.add:
            if curr_path_inf.inlist == 0 and curr_path_inf.prev >= 0:
                v2 = [[0.0, 0.0] for _ in range(TR_MAX_CAMS)]
                philf = [[PT_UNUSED] * MAX_CANDS for _ in range(TR_MAX_CAMS)]
                quali = assess_new_position(X[2], v2, philf, fb.buf[2], run_info)

                if quali >= 2:
                    X[3][:] = X[2]
                    in_volume = 0

                    v2_arr = np.array(v2[:fb.num_cams], dtype=np.float64)
                    X[3], dl = point_position(v2_arr, fb.num_cams, cpar.mm, cal)

                    if (vpar.X_lay[0] < X[3][0] < vpar.X_lay[1] and
                            run_info.ymin < X[3][1] < run_info.ymax and
                            vpar.Zmin_lay[0] < X[3][2] < vpar.Zmax_lay[1]):
                        in_volume = 1

                    diff_pos = X[2] - X[3]
                    if in_volume == 1 and pos3d_in_bounds(diff_pos, tpar):
                        angle, acc = angle_acc(X[1], X[2], X[3])
                        if ((acc < tpar.dacc and angle < tpar.dangle) or
                                (acc < tpar.dacc / 10)):
                            dl = (np.linalg.norm(X[1] - X[3]) +
                                  np.linalg.norm(X[0] - X[1])) / 2
                            rr = (dl / run_info.lmax + acc / tpar.dacc +
                                  angle / tpar.dangle) / quali
                            register_link_candidate(
                                curr_path_inf, rr, fb.buf[2].num_parts)
                            add_particle(fb.buf[2], X[3], philf)
                            num_added += 1
                    in_volume = 0

    for h in range(fb.buf[1].num_parts):
        curr_path_inf = fb.buf[1].path_info[h]
        if curr_path_inf.inlist > 0:
            sort(curr_path_inf.inlist, curr_path_inf.decis,
                 curr_path_inf.linkdecis)
            curr_path_inf.finaldecis = curr_path_inf.decis[0]
            curr_path_inf.next = curr_path_inf.linkdecis[0]

    for h in range(fb.buf[1].num_parts):
        curr_path_inf = fb.buf[1].path_info[h]
        if curr_path_inf.inlist > 0:
            ref_path_inf = fb.buf[2].path_info[curr_path_inf.next]
            if ref_path_inf.prev == PREV_NONE:
                ref_path_inf.prev = h
            else:
                if (fb.buf[1].path_info[ref_path_inf.prev].finaldecis >
                        curr_path_inf.finaldecis):
                    fb.buf[1].path_info[ref_path_inf.prev].next = NEXT_NONE
                    ref_path_inf.prev = h
                else:
                    curr_path_inf.next = NEXT_NONE
        if curr_path_inf.next != NEXT_NONE:
            count1 += 1

    print(f"step: {step}, curr: {fb.buf[1].num_parts}, "
          f"next: {fb.buf[2].num_parts}, links: {count1}, "
          f"lost: {fb.buf[1].num_parts - count1}, add: {num_added}")

    run_info.npart = run_info.npart + fb.buf[1].num_parts
    run_info.nlinks = run_info.nlinks + count1

    fb.fb_next()
    fb.write_frame_from_start(step)
    if step < run_info.seq_par.last - 2:
        fb.read_frame_at_end(step + 3, read_links=False)


def trackcorr_c_finish(run_info, step):
    range_val = run_info.seq_par.last - run_info.seq_par.first
    npart = run_info.npart / range_val
    nlinks = run_info.nlinks / range_val
    print(f"Average over sequence, particles: {npart:5.1f}, "
          f"links: {nlinks:5.1f}, lost: {npart - nlinks:5.1f}")

    run_info.fb.fb_next()
    run_info.fb.write_frame_from_start(step)


def trackback_c(run_info):
    from .orientation import point_position

    cal = run_info.cal
    seq_par = run_info.seq_par
    tpar = run_info.tpar
    vpar = run_info.vpar
    cpar = run_info.cpar
    fb = run_info.fb

    Ymin = 0.0
    Ymax = 0.0
    npart = 0.0
    nlinks = 0.0

    for step in range(seq_par.last, seq_par.last - 4, -1):
        fb.read_frame_at_end(step, read_links=True)
        fb.fb_next()
    fb.fb_prev()

    for step in range(seq_par.last - 1, seq_par.first, -1):
        for h in range(fb.buf[1].num_parts):
            curr_path_inf = fb.buf[1].path_info[h]

            if not ((curr_path_inf.next < 0) or (curr_path_inf.prev != -1)):
                continue

            X = [np.zeros(3) for _ in range(6)]
            curr_path_inf.inlist = 0
            X[1][:] = curr_path_inf.x

            ref_path_inf = fb.buf[0].path_info[curr_path_inf.next]
            X[0][:] = ref_path_inf.x
            X[2][:] = search_volume_center_moving(ref_path_inf.x, curr_path_inf.x)

            n = [[0.0, 0.0] for _ in range(fb.num_cams)]
            for j in range(fb.num_cams):
                n[j] = list(point_to_pixel(X[2], cal[j], cpar))

            w = sorted_candidates_in_volume(X[2], n, fb.buf[2], run_info)

            if w is not None:
                i = 0
                while w[i]['ftnr'] != TR_UNUSED:
                    ref_path_inf = fb.buf[2].path_info[w[i]['ftnr']]
                    X[3][:] = ref_path_inf.x

                    diff_pos = X[1] - X[3]
                    if pos3d_in_bounds(diff_pos, tpar):
                        angle, acc = angle_acc(X[1], X[2], X[3])
                        if ((acc < tpar.dacc and angle < tpar.dangle) or
                                (acc < tpar.dacc / 10)):
                            dl = (np.linalg.norm(X[1] - X[3]) +
                                  np.linalg.norm(X[0] - X[1])) / 2
                            quali = w[i]['freq']
                            rr = (dl / run_info.lmax + acc / tpar.dacc +
                                  angle / tpar.dangle) / quali
                            register_link_candidate(curr_path_inf, rr, w[i]['ftnr'])
                    i += 1

            if tpar.add:
                if curr_path_inf.inlist == 0:
                    v2 = [[0.0, 0.0] for _ in range(TR_MAX_CAMS)]
                    philf = [[PT_UNUSED] * MAX_CANDS for _ in range(TR_MAX_CAMS)]
                    quali = assess_new_position(X[2], v2, philf, fb.buf[2], run_info)
                    if quali >= 2:
                        in_volume = 0

                        v2_arr = np.array(v2[:fb.num_cams], dtype=np.float64)
                        X[3], _dl = point_position(v2_arr, fb.num_cams, cpar.mm, cal)

                        if (vpar.X_lay[0] < X[3][0] < vpar.X_lay[1] and
                                Ymin < X[3][1] < Ymax and
                                vpar.Zmin_lay[0] < X[3][2] < vpar.Zmax_lay[1]):
                            in_volume = 1

                        diff_pos = X[1] - X[3]
                        if in_volume == 1 and pos3d_in_bounds(diff_pos, tpar):
                            angle, acc = angle_acc(X[1], X[2], X[3])
                            if ((acc < tpar.dacc and angle < tpar.dangle) or
                                    (acc < tpar.dacc / 10)):
                                dl = (np.linalg.norm(X[1] - X[3]) +
                                      np.linalg.norm(X[0] - X[1])) / 2
                                rr = (dl / run_info.lmax + acc / tpar.dacc +
                                      angle / tpar.dangle) / quali
                                register_link_candidate(
                                    curr_path_inf, rr, fb.buf[2].num_parts)
                                add_particle(fb.buf[2], X[3], philf)
                        in_volume = 0

        for h in range(fb.buf[1].num_parts):
            curr_path_inf = fb.buf[1].path_info[h]
            if curr_path_inf.inlist > 0:
                sort(curr_path_inf.inlist, curr_path_inf.decis,
                     curr_path_inf.linkdecis)

        count1 = 0
        num_added = 0
        for h in range(fb.buf[1].num_parts):
            curr_path_inf = fb.buf[1].path_info[h]

            if curr_path_inf.inlist > 0:
                ref_path_inf = fb.buf[2].path_info[curr_path_inf.linkdecis[0]]

                if (ref_path_inf.prev == PREV_NONE and
                        ref_path_inf.next == NEXT_NONE):
                    curr_path_inf.finaldecis = curr_path_inf.decis[0]
                    curr_path_inf.prev = curr_path_inf.linkdecis[0]
                    fb.buf[2].path_info[curr_path_inf.prev].next = h
                    num_added += 1

                if (ref_path_inf.prev != PREV_NONE and
                        ref_path_inf.next == NEXT_NONE):
                    X = [np.zeros(3) for _ in range(6)]
                    X[0][:] = fb.buf[0].path_info[curr_path_inf.next].x
                    X[1][:] = curr_path_inf.x
                    X[3][:] = ref_path_inf.x
                    X[4][:] = fb.buf[3].path_info[ref_path_inf.prev].x
                    for j in range(3):
                        X[5][j] = 0.5 * (5.0 * X[3][j] - 4.0 * X[1][j] + X[0][j])

                    angle, acc = angle_acc(X[3], X[4], X[5])
                    if ((acc < tpar.dacc and angle < tpar.dangle) or
                            (acc < tpar.dacc / 10)):
                        curr_path_inf.finaldecis = curr_path_inf.decis[0]
                        curr_path_inf.prev = curr_path_inf.linkdecis[0]
                        fb.buf[2].path_info[curr_path_inf.prev].next = h
                        num_added += 1

            if curr_path_inf.prev != PREV_NONE:
                count1 += 1

        print(f"step: {step}, curr: {fb.buf[1].num_parts}, "
              f"next: {fb.buf[2].num_parts}, links: {count1}, "
              f"lost: {fb.buf[1].num_parts - count1}, add: {num_added}")

        npart = npart + fb.buf[1].num_parts
        nlinks = nlinks + count1

        fb.fb_next()
        fb.write_frame_from_start(step)
        if step > seq_par.first + 2:
            fb.read_frame_at_end(step - 3, read_links=True)

    npart /= (seq_par.last - seq_par.first - 1)
    nlinks /= (seq_par.last - seq_par.first - 1)

    print(f"Average over sequence, particles: {npart:5.1f}, "
          f"links: {nlinks:5.1f}, lost: {npart - nlinks:5.1f}")

    fb.fb_next()
    fb.write_frame_from_start(step)

    return nlinks
