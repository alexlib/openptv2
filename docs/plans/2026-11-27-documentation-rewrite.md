# Documentation rewrite: a slow course for beginners, and recipes by situation

## The problem, in plain words

The docs today are written by the people who built the code, for people who already understand it. Pages mix three things in
one text: what a step means, how to run it, and what we found out while building it. A new user cannot tell where to start,
cannot run the steps (most examples need data that is not in the repository), and cannot tell whether the result is right.
Some pages are named after one dataset (`illmenau-4cam-calibration.md`), so a user with a different rig does not look there.

## What we build

Two separate parts, and a third that keeps the existing deep material out of the way.

### Part A. Learn (the basics, slow, runnable)

One numbered course. Every page has the same shape: **what this step does** (3 sentences, no jargon) → **what you need** →
**do it** (one command or a few clicks, copy-paste) → **what you should see** (the real output or picture) → **check it
worked** (a number or a picture with a pass/fail rule) → **if it did not** (the 3 commonest causes) → **next**.

It runs on data that ships with the repository (`test_data/test_cavity_small`, 3 MB, 4 cameras, so a reader can type every
command and compare with the page). Each page has a marimo notebook next to it where the idea is interactive (sliders),
and the whole course is exercised by one smoke test, so a page cannot drift from the code unnoticed.

| # | Page | What the reader learns and can check |
|---|---|---|
| 0 | Start here | What 3D-PTV measures, what you need (cameras, calibration, particles), the 5 stages in one picture, a glossary |
| 1 | Install and verify | One command prints "OK" and the compiled/interpreted mode |
| 2 | Look at your data | Folder layout, image naming, the YAML file, what each key means; `openptv2-check` on a folder |
| 3 | Cameras and calibration | What a camera model is (position, angles, focal length, lens distortion), what `.ori` / `.addpar` contain, what the glass window does (uses the ray-tracing notebook) |
| 4 | Find the particles | Threshold, blob size, what a good detection looks like on an image, how to tune on one frame |
| 5 | Match across cameras | Epipolar lines in a picture, what `eps0` means, how many 4-camera points to expect, ghosts |
| 6 | Track in time | Predict-and-link in a picture, the default tracker, what a good track looks like |
| 7 | Is the result good? | The 4 checks (tracks per frame, track lengths, smoothness, leave-one-camera-out) with pass/fail numbers |
| 8 | Is the calibration good? | Reprojection error vs ray convergence, the camera-switch jump, `consistency_report`, when to self-calibrate and when not |
| 9 | Use the results | Reading the output (Zarr/HDF5/text), velocities, plotting, exporting |

### Part B. Recipes (by what you have, not by whose dataset it was)

Each recipe starts with "You have ... You want ...", lists the exact inputs, and links to the basics for each step instead
of repeating them. Names describe the situation.

| Recipe | Replaces / uses |
|---|---|
| Two rigs of four cameras each, calibrated with a hand-held plate (OpenCV-style free pose) | `illmenau-4cam-calibration.md`, `illmenau-pipeline.md`, `plate-calibration-howto.md` |
| One rig, a plate or target held at known positions (the classic multi-plane calibration) | `tutorials/calibration.md` |
| Cameras on opposite sides of a water cell (cavity) | cavity pages, `test-cavity` notes, synthetic cavity study |
| I already have a calibration: is it good enough? | calibration consistency, RMS vs RCM |
| Tracks break when the set of cameras changes | self-calibration, scale bias, layer 3 alignment |
| Dense seeding, merged blobs | shared-blob material |
| Many runs sharing one calibration; running in the cloud | `multi-folder-runs.md`, `cloud-batch.md` |
| Calibrating with a dumbbell | `tutorials/dumbbell_calibration.md` |

### Part C. Reference and research (kept, moved out of the way)

The long pages (bundle adjustment, scale bias, benchmark results, synthetic studies, plans) stay, under "Reference and
research", each starting with a 5-line summary and a link to the basic page it extends. Nothing is deleted.

## Rules for every page (a checklist reviewers use)

1. A reader who has never heard of the code can finish it in the stated time. Test: someone else follows it cold.
2. Words: one sentence per idea; every term explained the first time (link to the glossary); code names in parentheses.
3. Every command has the output it should print, copied from a real run.
4. Every number has a rule ("good: below 0.3 px; look again if above 1 px").
5. Dataset names do not appear in titles or file names; they appear as "example: ...".
6. Pictures first where a picture explains it (epipolar lines, the window, tracks); notebooks where a slider teaches more than text.
7. No research findings inside a how-to; a one-line link to the reference page instead.

## Verification

* `uv run mkdocs build --strict` passes (no broken links, nothing outside the navigation by accident).
* `tests/docs/test_course_smoke.py` runs every command of Part A on `test_cavity_small` and compares key numbers with the
  page (a tolerance), so a change in the code that breaks the course fails CI.
* Cold read: for each Part A page and each recipe, a fresh reader (a subagent with no context, then the user) follows it and
  lists every place they stopped; those places are fixed before the page is accepted.
* Old URLs keep working (`mkdocs-redirects`) when a page is renamed.

## Order of work (each step reviewable)

1. **Skeleton and rules** (1 day): navigation with the three parts, the page template, glossary, the smoke test harness, redirects.
2. **Part A pages 0-3** (install, data, cameras): the foundation, including the ray-tracing notebook linked from page 3.
3. **Part A pages 4-7** (detect, match, track, check) with figures made from the bundled data.
4. **Part A pages 8-9** (calibration check, use results).
5. **Recipe 1** (two 4-camera rigs with plates), rewritten from the Illmenau material, plus renaming scripts with the old names kept as aliases.
6. **Recipes 2-8**, one at a time, each from the existing material.
7. **Part C**: summaries added to the long pages, nav moved.
8. Cold-read pass over everything; fix; publish.

## Decisions needed from the user

1. **Course data:** `test_cavity_small` ships in the repository (3 MB) and is the only 4-camera set small enough. Is it acceptable
   as the course dataset, or should I generate a small synthetic one with a known answer (the truth rig), which also lets the
   reader see an error against the truth?
2. **GUI or command line first?** The current docs lead with the GUI. Proposal: each step shows the command line and the GUI
   route side by side, command line first (it is the one we can test in CI).
3. **Renames:** scripts and the environment variable carry the dataset name (`calibrate_illmenau_4cam.py`,
   `ILLMENAU_NO_HINT`). Proposal: new neutral names, old names kept for one release as aliases (same policy as retired code).
4. **Language of the recipe title:** "Two rigs of four cameras, calibrated with plates" is the working title; say if you prefer another.
