import os

import numpy as np
from numba import njit
import h5py


"""
Main class for computing field lines
"""


class FieldLines:
    """
    Reads data from a .hdf5 file and computes field lines
    """

    def __init__(self, data_path):

        # Get array of shape (S, N, 3) with positions
        with h5py.File(data_path, "r") as file:
            S = file["Header"].attrs["num_snapshots"]  # number of time steps
            N = file["Header"].attrs["num_particles"][0]  # number of particles
            e = file["Header"].attrs["SofteningLength"]
            masses = file["Masses"]["Masses"][()]
            positions = np.zeros((S, N, 3))
            for i in range(S):
                positions[i] = file[f"{i:03d}"]["Positions"]

        self.positions = positions
        self.masses = masses
        self.e = e
        self.G = 1

        self.S = S
        self.N = N

        self.meshgrid_defined = False

    def save_field_lines(self, savefold, zs, t_idxs, vals, filename="field_lines"):
        """

        Compute and save equipotential lines for a given set of z coordinate values, time steps and potential values
        Saves the data in a .hdf5 file

        :zs: list of z values where to compute the equipotential lines
        :t_idxs: list of time steps where to compute the equipotential lines
        :vals: list of values of the potential to compute the equipotential lines
        """

        # Check if savefold exists and if not create it
        if not os.path.exists(savefold):
            os.makedirs(savefold)

        if isinstance(t_idxs, int):
            t_idxs = [t_idxs]
        if isinstance(zs, int) or isinstance(zs, float):
            zs = [zs]
        if isinstance(vals, int) or isinstance(vals, float):
            vals = [vals]

        savepath = f"{savefold}/{filename}.hdf5"
        with h5py.File(savepath, "w") as f:
            header_grp = f.create_group("Header")
            header_grp.attrs["Nparticles"] = np.int32(self.N)
            header_grp.attrs["Nsteps"] = np.int32(len(t_idxs))
            header_grp.attrs["Lims"] = np.array(self.lim, dtype=np.float32)
            header_grp.attrs["IsoValues"] = np.array(vals, dtype=np.float32)
            header_grp.attrs["Zs"] = np.array(zs, dtype=np.float32)

        with h5py.File(savepath, "a") as f:
            for i, t_idx in enumerate(t_idxs):
                for j, val in enumerate(vals):

                    # We need to be careful with the indexing here, because some values of z do not have any equipotential lines for a given value of the potential
                    ks = []
                    for k, z in enumerate(zs):
                        equilines = self.get_equipotential_lines(val, z, t_idx)
                        N_loops = len(equilines)
                        non_empty_equilines = []
                        for l in range(N_loops):
                            equiline = equilines[l]
                            z_arr = np.zeros((equiline.shape[0], 1)) + z
                            equiline = np.concatenate((equiline, z_arr), axis=1)
                            non_empty_equilines.append(equiline)
                            if k not in ks:
                                ks.append(k)

                        group = f.create_group(f"{t_idx:03d}_{j:03d}_{k:03d}")

                        for l, equiline in enumerate(non_empty_equilines):
                            group.create_dataset(
                                f"{t_idx:03d}_{j:03d}_{k:03d}_{l:03d}",
                                data=equiline,
                                dtype=np.float32,
                            )

                    f.create_dataset(
                        f"{t_idx:03d}_{j:03d}_ks", data=np.array(ks), dtype=np.int32
                    )

                print(f"\r{i/(len(t_idxs)-1)*100:.2f}%", end="")

        self.savepath = savepath

    def get_equipotential_lines(self, val, z, j):

        self.get_potential_grid(z, j)
        equilines = marching_square(self.x, self.y, self.PHI, val)

        equilines = get_loops(equilines)

        return equilines

    def define_meshgrid(self, lim, M):

        if isinstance(lim, int) or isinstance(lim, float):
            lim = np.array([-lim, lim])

        self.lim = lim
        self.M = M

        self.x = np.linspace(lim[0], lim[1], M)
        self.y = np.linspace(lim[0], lim[1], M)

        self.meshgrid_defined = True

    def get_potential_grid(self, z, j, prints=False):

        if not self.meshgrid_defined:
            print("Meshgrid not defined!")
            return

        X, Y = np.meshgrid(self.x, self.y)

        self.PHI = self._potential(X, Y, z, j)

        if prints:
            print(f"PHI_min = {np.min(self.PHI)}")
            print(f"PHI_max = {np.max(self.PHI)}")

    def _potential(self, x, y, z, j):
        """
        Total potential at (x, y, z) at time-step j
        """

        U = 0
        for i in range(self.N):
            x_body = self.positions[j, i, 0]
            y_body = self.positions[j, i, 1]
            z_body = self.positions[j, i, 2]
            U += (
                self._potential_over_m(x, y, z, x_body, y_body, z_body) * self.masses[i]
            )
        return U

    def _potential_over_m(self, x, y, z, x_body, y_body, z_body):

        dx = x - x_body
        dy = y - y_body
        dz = z - z_body

        d = np.sqrt(dx**2 + dy**2 + dz**2 + self.e**2)
        return -self.G / d


"""
Utility functions for Marching Squares, optimized with numba
"""


@njit
def compute_case_id(data, threshold):
    rows, cols = data.shape[0] - 1, data.shape[1] - 1
    case_id = np.zeros((rows, cols), dtype=np.int32)
    for j in range(rows):
        for i in range(cols):
            case_id[j, i] = (
                (data[j + 1, i] >= threshold) * 1
                | (data[j + 1, i + 1] >= threshold) * 2
                | (data[j, i + 1] >= threshold) * 4
                | (data[j, i] >= threshold) * 8
            )
    return case_id


@njit
def marching_square(xVector, yVector, data, threshold):
    case_id = compute_case_id(data, threshold)
    rows, cols = case_id.shape
    max_lines = (
        rows * cols * 2
    )  # Estimate of max number of lines, assuming at most 2 per cell
    lines = np.zeros((max_lines, 4))  # Each line is (pX, pY, qX, qY)
    line_index = 0

    for j in range(rows):
        for i in range(cols):
            A_x, A_y = xVector[i], yVector[j + 1]
            B_x, B_y = xVector[i + 1], yVector[j + 1]
            C_x, C_y = xVector[i + 1], yVector[j]
            D_x, D_y = xVector[i], yVector[j]

            mid_AB_x = (A_x + B_x) / 2
            mid_CD_x = (C_x + D_x) / 2
            mid_AD_y = (A_y + D_y) / 2
            mid_BC_y = (B_y + C_y) / 2

            if case_id[j, i] in (1, 14, 10):
                lines[line_index] = (mid_AB_x, B_y, D_x, mid_AD_y)
                line_index += 1
            if case_id[j, i] in (2, 13, 5):
                lines[line_index] = (mid_AB_x, A_y, C_x, mid_AD_y)
                line_index += 1
            if case_id[j, i] in (3, 12):
                lines[line_index] = (A_x, mid_AD_y, C_x, mid_BC_y)
                line_index += 1
            if case_id[j, i] in (4, 11, 10):
                lines[line_index] = (mid_CD_x, D_y, B_x, mid_BC_y)
                line_index += 1
            if case_id[j, i] in (6, 9):
                lines[line_index] = (mid_AB_x, A_y, mid_CD_x, C_y)
                line_index += 1
            if case_id[j, i] in (7, 8, 5):
                lines[line_index] = (mid_CD_x, C_y, A_x, mid_AD_y)
                line_index += 1

    return lines[:line_index]


"""
Marching squares returns a collection of segments for each equipotential line. 
The funcion get_loops takes these segments and returns a list of numpy arrays, each one representing a closed loop.
"""


def find_connected_points(lines):
    connections = {}
    for x1, y1, x2, y2 in lines:
        point1 = (x1, y1)
        point2 = (x2, y2)
        if point1 in connections:
            connections[point1].append(point2)
        else:
            connections[point1] = [point2]
        if point2 in connections:
            connections[point2].append(point1)
        else:
            connections[point2] = [point1]
    return connections


def trace_loop(start_point, connections):
    loop = []
    current_point = start_point
    previous_point = None
    while True:
        loop.append(current_point)
        next_points = connections[current_point]
        if len(next_points) == 1:
            # Only one connection means this is either an end point or a starting point in an open line
            next_point = next_points[0]
        else:
            # Find the next point that is not the previous one
            next_point = (
                next_points[0] if next_points[1] == previous_point else next_points[1]
            )
        if next_point == start_point or next_point == previous_point:
            break
        previous_point, current_point = current_point, next_point
    return loop


def get_loops(lines):
    connections = find_connected_points(lines)
    loops = []
    visited = set()
    for start_point in connections:
        if start_point not in visited:
            loop = trace_loop(start_point, connections)
            loops.append(np.array(loop))
            visited.update(loop)
    return loops
