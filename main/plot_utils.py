import os
import subprocess
import time
import shutil

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class Figure3D:
    def __init__(
        self,
        fig_size=540,
        ratio=2,
        dpi=300,
        ts=1.7,
        pad=0.2,
        pad_color="#454545",
        color="k",
        ax_color="w",
        azimuth=25,
        polar=45,
        lims=1,
        show=True,
    ):

        fig_width, fig_height = fig_size * ratio / dpi, fig_size / dpi
        fs = np.sqrt(fig_width * fig_height)

        self.fs = fs
        self.fig_size = fig_size
        self.ratio = ratio
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.ts = ts
        self.pad = pad
        self.dpi = dpi
        self.color = color
        self.ax_color = ax_color
        self.pad_color = pad_color

        self.show = show

        self.fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

        self.fig.patch.set_facecolor(pad_color)

        ax = self.fig.add_subplot(111, projection="3d")
        self.ax = ax

        ax.view_init(elev=azimuth, azim=polar)

        ax.set_facecolor(color)

        if isinstance(lims, (int, float)):
            lims = [(-lims, lims)] * 3
        if isinstance(lims, (list, tuple)):
            for i, lim in enumerate(lims):
                if isinstance(lim, (int, float, np.float32, np.int32)):
                    lims[i] = (-lim, lim)

        self.x_lim = lims[0]
        self.y_lim = lims[1]
        self.z_lim = lims[2]

        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_zlim(self.z_lim)

        for k, axis in enumerate([ax.xaxis, ax.yaxis, ax.zaxis]):
            axis.line.set_linewidth(0 * fs)
            axis.set_visible(False)
            axis.pane.fill = False
            axis.pane.set_edgecolor("none")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        dx = lims[0][1] - lims[0][0]
        dy = lims[1][1] - lims[1][0]
        dz = lims[2][1] - lims[2][0]

        ax.set_box_aspect([1, dy / dx, dz / dx])

    def save(self, path, bbox_inches="tight", pad_inches=0, show=False):

        self.fig.savefig(
            path, dpi=self.dpi, bbox_inches=bbox_inches, pad_inches=pad_inches
        )

        self.path = path

        if not show:
            plt.close(self.fig)


def frames_to_mp4(
    fold,
    title="video",
    fps=36,
    digit_format="04d",
    res=None,
    resize_factor=1,
    custom_bitrate=None,
    extension=".jpg",
):

    # Get a list of all .png files in the directory
    files = [f for f in os.listdir(fold) if f.endswith(extension)]
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    if not files:
        raise ValueError("No image files found in the specified folder.")

    basename = os.path.splitext(files[0])[0].split("_")[0]

    ffmpeg_path = "ffmpeg"
    abs_path = os.path.abspath(fold)
    parent_folder = os.path.dirname(abs_path) + os.sep
    output_file = os.path.join(parent_folder, f"{title}.mp4")

    crf = 5  # Lower for higher quality, higher for lower quality
    bitrate = custom_bitrate if custom_bitrate else "5000k"
    preset = "slow"
    tune = "film"

    command = f'{ffmpeg_path} -y -r {fps} -i {os.path.join(fold, f"{basename}_%{digit_format}{extension}")} -c:v libx264 -profile:v high -crf {crf} -preset {preset} -tune {tune} -b:v {bitrate} -pix_fmt yuv420p -vf {output_file}'

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error during video conversion:", e)


def frames_to_gif(
    fold,
    title="video",
    outfold=None,
    fps=24,
    digit_format="04d",
    quality=500,
    max_colors=256,
    extension=".jpg",
):

    files = [f for f in os.listdir(fold) if f.endswith(extension)]
    files.sort()

    name = os.path.splitext(files[0])[0]
    basename = name.split("_")[0]

    ffmpeg_path = "ffmpeg"
    framerate = fps

    if outfold is None:
        abs_path = os.path.abspath(fold)
        parent_folder = os.path.dirname(abs_path) + "\\"
    else:
        parent_folder = outfold
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)

    output_file = parent_folder + "{}.gif".format(title)

    # Create a palette with limited colors for better file size
    palette_file = parent_folder + "palette.png"
    palette_command = f'{ffmpeg_path} -i {fold}{basename}_%{digit_format}{extension} -vf "fps={framerate},scale={quality}:-1:flags=lanczos,palettegen=max_colors={max_colors}" -y {palette_file}'
    subprocess.run(palette_command, shell=True)

    # set paletteuse
    paletteuse = "paletteuse=dither=bayer:bayer_scale=5"

    # Use the optimized palette to create the GIF
    gif_command = f'{ffmpeg_path} -r {framerate} -i {fold}{basename}_%04d{extension} -i {palette_file} -lavfi "fps={framerate},scale={quality}:-1:flags=lanczos [x]; [x][1:v] {paletteuse}" -y {output_file}'
    subprocess.run(gif_command, shell=True)
