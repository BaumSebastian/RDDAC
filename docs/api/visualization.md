# Visualization

Matplotlib plotting helpers that operate on numpy arrays. Pair them with `rddac.open_h5` to read the input arrays from a single experiment. See the [Visualization tutorial](../tutorials/visualization.md) for end to end examples.

```python
import rddac

with rddac.open_h5(0) as f:
    z     = f["pointcloud/op10/z"][:]
    lumi  = f["pointcloud/op10/luminescence"][:]
    force = f["force/data"][:]

ax, cbar = rddac.plot_scan(z)                       # heightmap image
pts      = rddac.scan_to_pointcloud(z, lumi)        # flat buffer -> (N, 3)
ax, cbar = rddac.plot_point_cloud(pts)
ax       = rddac.plot_force(force)                  # force time series
```

## Functions

::: rddac.visualization.plot_scan

::: rddac.visualization.scan_to_pointcloud

::: rddac.visualization.plot_point_cloud

::: rddac.visualization.plot_force

::: rddac.visualization.plot_traverse

## Constants

```python
rddac.visualization.SCAN_X_SHAPE = 3200      # scan grid width  [px]
rddac.visualization.SCAN_Y_SHAPE = 2000      # scan grid height [px]

rddac.visualization.FORCE_COLUMNS = (
    "time", "load_cell_1", "load_cell_2", "load_cell_3", "load_cell_4",
    "punch_temp", "punch_pos", "total_force",
)
```
