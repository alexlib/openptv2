# 2. Look at your data: images, folders, and the parameter file

**Time:** 10 minutes. **You need:** the repository folder from the previous page.

## What this step does

Before computing anything, you should know what the program expects to find: which files, in which folders, with which names,
and where the settings live. Almost every problem a beginner meets is a file in the wrong place or a setting that does not
match the images.

## The example

`test_data/test_cavity_small` is a real experiment cut down to 5 frames and 256 x 256 pixel images:

```text
test_cavity_small/
├── parameters_Run1.yaml      all the settings of this experiment (one file)
├── img/                      the images: one file per camera and frame
│   ├── cam1.10000   cam1.10001 ... cam1.10004
│   ├── cam2.10000 ...   cam3.10000 ...   cam4.10000 ...
├── cal/                      the calibration: two small text files per camera
│   ├── cam1.tif.ori          where camera 1 is, and how it looks (next page)
│   ├── cam1.tif.addpar       its lens distortion
│   └── ... cam2, cam3, cam4
└── res/                      results will be written here
```

* **Image names** are `cam<camera number>.<frame number>`. The frame number is a counter; here 10000 to 10004 (the YAML says
  so under `sequence: first / last`).
* **One frame** is one moment in time, seen by all four cameras: `cam1.10000`, `cam2.10000`, `cam3.10000`, `cam4.10000`.

![The four cameras' images of frame 10000](figures/four_images.png)

*The same moment, seen from four sides. The bright dots are tracer particles. Note that the four pictures are different: each camera sees the volume from another side.*

## The parameter file

`parameters_Run1.yaml` holds every setting. You will rarely edit it by hand (the interface does it for you), but you must be
able to read it. The parts that matter first:

| Section | Key | Example | Meaning |
|---|---|---|---|
| (top) | `num_cams` | 4 | number of cameras |
| `ptv` | `imx`, `imy` | 256, 256 | image size in pixels |
| `ptv` | `pix_x`, `pix_y` | 0.012 | size of one sensor pixel in mm (12 micrometres) |
| `ptv` | `mmp_n1`, `mmp_n2`, `mmp_n3`, `mmp_d` | 1.0, 1.46, 1.33, 6.0 | refractive index of air, glass, water, and the glass thickness in mm (page 3) |
| `cal_ori` | `img_ori` | `cal/cam1.tif.ori` ... | which calibration file belongs to which camera |
| `sequence` | `first`, `last` | 10000, 10004 | the frames to process |
| `criteria` | `X_lay`, `Zmin_lay`, `Zmax_lay` | -10..10, -20..25 | the box in mm where you expect particles (the measurement volume) |
| `criteria` | `eps0` | 0.06 | how far (mm on the sensor) a spot may be from its epipolar line and still match (page 5) |
| `track` | `dvxmax`... | 15.5 | the largest step a particle may make between two frames, mm |

The sizes and the volume are the settings that most often do not match the user's own images.

## Do it

=== "Command line"

    Print the settings the program reads, to be sure it finds the files:

    ```bash
    python -c "
    from openptv2.recorrespond import load_parameters
    cpar, vpar, cals, params = load_parameters('test_data/test_cavity_small/parameters_Run1.yaml')
    print('cameras:', len(cals), ' image:', cpar.imx, 'x', cpar.imy, ' pixel:', cpar.pix_x, 'mm')
    "
    ```

=== "Graphical interface"

    ```bash
    openptv2-gui -w test_data/test_cavity_small/parameters_Run1.yaml
    ```

    In the menu: **Start → Init / Reload**. The four camera windows fill with the images of the first frame. If a window stays
    empty, the image path in the YAML is wrong.

## What you should see

Four lines of *Loaded calibration for camera N from ...*, then

```text
cameras: 4  image: 256 x 256  pixel: 0.012 mm
```

## Check that it worked

The size and the number of cameras printed must match what you know about your own images. If you point the program at your own
folder and these numbers are wrong, nothing after this will be right.

## If it did not work

| You see | Cause | Do |
|---|---|---|
| `FileNotFoundError` naming a `.ori` | the path in `cal_ori` is relative to the folder you started from | run from the repository folder, or give the full path |
| empty camera windows | image names do not match `cam<N>.<frame>` | rename, or change `base_name` under `sequence` |
| wrong `cameras:` number | `num_cams` and the number of `img_ori` entries differ | make them agree |

[Next: cameras and calibration →](03-cameras.md){ .md-button .md-button--primary }
