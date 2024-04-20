![Example](figures/animation.gif)

Animation of equipotential surfaces on a three-body periodic orbit

## Structure

- `data/`: Folder with output data in `.hdf5` format from a 3-body simulation 
- `main/`
    - `main/equilines.py`: Main functionality in the FieldLines class, which takes directly the simulation output data and computes and saves field lines using a marching squares algorithm.
    - `main/plot_utils.py`: Utility functions for plotting
    - `main/utils.py`: Contains a single function for the animation,
- `main_notebook.ipynb`: Notebook where the animation is done

## Usage

Check  `main_notebook.ipynb` for an example.

## Requirements

`numpy`, `matplotlib`, `h5py`, `PIL` and `numba` 
