# covid-19_shapecutter
Met Office COVID-19 response: Cutting gridded data with shapefiles

## Shapecutter example notebook
This repo contains an example of how to subset a gridded NetCDF file with a shapefile polygon using:
- [Iris](https://scitools.org.uk/iris/docs/latest/userguide/index.html) for data loading, subsetting and statistical processing
- [Cartopy](https://scitools.org.uk/cartopy/docs/latest/tutorials/using_the_shapereader.html) for loading and reading shapefiles
- [`shape_utils.py`](shape_utils.py) for using shapefiles to subset an Iris cube.
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) to collate the data into a tabular form and save out to a Comma Separated Vector (CSV) file.

You will need [Jupyter](https://jupyter.org/install) installed in order to use this notebook.

## Cloning this repository
You can clone this repository using git:

```git clone https://github.com/informatics-lab/covid-19_shapecutter.git```

## Software Environment
You an use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to install the right software dependencies, including Jupyter, as directed in the `environment.yaml`:

```conda env create --file environment.yaml```
