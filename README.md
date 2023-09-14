# MSc Individual Project

## Setup

-   Install dependencies requirements.txt
-   Store raw 3D Muliti-echo GRE data as follows:
    Data/\
    &emsp;+---invivo/\
    &emsp;&emsp;+---p01/\
    &emsp;&emsp;| &emsp;+---f01/\
    &emsp;&emsp;| &emsp;| &emsp;+---image.cfl\
    &emsp;&emsp;| &emsp;| &emsp;+---image.hdr\
    &emsp;&emsp;| &emsp;+---f02/\
    &emsp;&emsp;| &emsp;...\
    &emsp;&emsp;+---p02/\
    &emsp;&emsp;...\

## Field Estimator Demos

-   $ python3 -m FieldEstimator phase_fitting_demo
-   $ python3 -m FieldEstimator delat_omega_maps
-   $ python3 -m FieldEstimator B0_hist

## Simulator Demos

-   $ python3 -m Simulator analytic_vs_numerical
-   $ python3 -m Simulator generate_analytic_residuals
-   $ python3 -m Simulator generate_numerical_residuals
-   $ python3 -m Simulator single_voxel_vsf
-   $ python3 -m Simulator compare_interpolations

## StarMap Demos

-   $ python3 -m StarMap pixel_demo
-   $ python3 -m StarMap loss_curve

## Pipeline demos (Integration of above algorithms)

-   $ python3 -m Pipeline reconstructions
-   $ python3 -m Pipeline pixel_reconstructions
-   $ python3 -m Pipeline speed_and_residuals
-   $ python3 -m Pipeline param_map_demo
