Multivariate Gaussian Distribution Generator
===

This program reads JSON formatted data, formats this dataset for use
with gaussians, and calculates the covariance matrix (and mean). It
then outputs a sampled region of the data to the console.

Take care, all parameters (e.g. which region to sample and what the
data format is) are hardcoded in the `main.cpp` file. All other files
are trimmed to be used dynamically.


Dependencies
---

This program depends on

 * Eigen3
 * JSON-C

Install them using

```bash
sudo apt-get install libeigen3-dev libjson-c-dev
```

Usage Semantics
---

A sample dataset is included with this, located in
`data/grasp_positions.json`. The data format is `JSON`, with each line
representing a single JSON document. The data in each document is
structured as follows:

```
[success: string, x: double, y: double, z: double]
```

The data describes the `x`/`y`/`z` positions of a robot grasping an
object, with success being denoted as `"True"` or `"False"`.

In the `main.cpp` file, `z` is ignored as its value is always `0`;
otherwise, the covariance matrix won't have full rank, making
calculating the Gaussian from it impossible.

The output of the program is a `CSV` formatted list of sampled points
in `x`/`y` space, followed by the probability of successfully grasping
the object. The output on each line looks like this:

```
x, y, p
```

Compiling it
---

Compile it like you would with any other CMake-based project:

```bash
mkdir build
cd build
cmake ..
make
```


Running it
---

A set of scripts is prepared to run the program, process its output,
and plot it using `gnuplot`. Install `gnuplot` using

```bash
sudo apt-get install gnuplot
```

Run the program in its root directory by doing this:

```bash
./scripts/run.sh
```

It generated the file `heatmap.pdf` in the project's root
directory. View it with your favourite PDF viewer.
