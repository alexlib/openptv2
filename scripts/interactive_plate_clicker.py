import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from openptv2.detect_plate import detect_plate_targets, plate_tpar_from_yaml
from openptv2.plate_labeler import label_coded_6x7, label_uncoded_grid
from openptv2.algorithms.parameters import ControlPar, MmNp

def main():
    parser = argparse.ArgumentParser(description='Interactive Plate Grid Clicker')
    parser.add_argument('--base', type=str, default=r'C:\Users\alex\Downloads\Illmenau')
    parser.add_argument('--cam', type=int, default=1, help='Camera index 1..4')
    parser.add_argument('--frame', type=str, default='00000000', help='Frame prefix')
    parser.add_argument('--pitch', type=float, default=120.0, help='Grid pitch mm')
    args = parser.parse_args()
    cam_folder = Path(args.base) / ('Kalibrierung_' + str(args.cam))
    tifs = list(cam_folder.glob(args.frame + '*.tiff')) + list(cam_folder.glob(args.frame + '*.tif'))
    if not tifs:
        print('No image found')
        return
    img = Image.open(tifs[0])
    raw = np.array(img)
    yaml_path = Path(args.base) / 'openptv_illmenau_4cam' / 'parameters_Run1.yaml'
    tpar = plate_tpar_from_yaml(yaml_path) if yaml_path.exists() else None
    cpar = ControlPar(num_cams=4, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005, mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0), chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0, img_base_name=['']*4, cal_img_base_name=['']*4)
    res = detect_plate_targets(raw, tpar, cpar, cam=args.cam - 1, coded_thr=30.0)
    centroids = res.centroids
    print('Detected dots:', len(centroids))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(raw, cmap='gray', origin='upper')
    if len(centroids) > 0:
        ax.scatter(centroids[:, 0], centroids[:, 1], s=25, facecolors='none', edgecolors='cyan', label='Detected Dots')
    clicked = []
    scat = ax.scatter([], [], s=80, c='red', marker='x', label='Clicked Anchors')
    ax.set_title('Cam ' + str(args.cam) + ': Click 3 Coded L-Dots (Corner, +1Y, +2X) or 4 Corners. Close when done.')
    ax.legend(loc='upper right')
    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        xy = np.array([event.xdata, event.ydata])
        if len(centroids) > 0:
            snapped = centroids[np.argmin(np.linalg.norm(centroids - xy, axis=1))]
        else:
            snapped = xy
        clicked.append(snapped)
        print('Anchor', len(clicked), 'snapped to', snapped)
        scat.set_offsets(np.array(clicked))
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
