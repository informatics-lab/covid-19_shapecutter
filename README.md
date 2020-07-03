# covid-19_shapecutter
Met Office COVID-19 response: Cutting gridded data with shapefiles

## Shapecutter example notebook
This repo contains an example and script to subset a gridded NetCDF file with a shapefile polygon using:
- [Iris](https://scitools.org.uk/iris/docs/latest/userguide/index.html) for data loading, subsetting and statistical processing
- [Cartopy](https://scitools.org.uk/cartopy/docs/latest/tutorials/using_the_shapereader.html) for loading and reading shapefiles
- [`shape_utils.py`](shape_utils.py) for using shapefiles to subset an Iris cube.
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) to collate the data into a tabular form and save out to a Comma Separated Vector (CSV) file.

You will need [Jupyter](https://jupyter.org/install) installed in order to use `shapecutter_notebook.ipynb`. <br>(See [Software Environment](#software-environment) for dependency installation guidelines.)

## Cloning this repository
You can clone this repository using git in a terminal:

```
$ git clone https://github.com/informatics-lab/covid-19_shapecutter.git
```

## Run *shapecutter_notebook.ipynb* using Jupyter
### Pangeo platform
If you have cloned this reposiory onto a platform like [Pangeo](https://covid19-response.informaticslab.co.uk) you can simply open `shapecutter_notebook.ipynb` and select the `covid-19` kernel and start running the cells.

### Local Jupyter instance
You can run `shapecutter_notebook.ipynb` locally either as a standalone Jupyter Notebook:
```
$ jupyter notebook shapecutter_notebook.ipynb
```

Or in a Jupyter Lab session:
```
$ jupyter lab shapecutter_notebook.ipynb
```

## Run *shapecutter_script.py* in a terminal
You can run `shapecutter_script.py` in a terminal session, provided you have the right conda environment running.

### Running a conda environment
```
$ conda activate shapecutter
```
This will ensure the software libraries that `shapecutter_script.py` depends on are available. <br>
This only needs to be ran once per terminal session. <br>
See [Software Environment](#software-environment) for instructions on creating the right conda environment.

### Run with default parameters
```
$ python shapecutter_script.py 20200601
```

This will run the script for the data on date 01-June-2020, using default values for the other parameters. <br>To change the default parameters in the script, edit `shapecutter_script.py` at the [following section](https://github.com/informatics-lab/covid-19_shapecutter/blob/master/shapecutter_script.py#L221):
```python
if __name__ == "__main__":

    # DEFAULT VALUES
    DATA_FOLDER = '/data/met-office/open-license/metoffice_global_daily/'
    SKIP_FILES = []
    FILENAME_FORMAT = '*_%Y%m%d.nc'
    SHAPEFILE = '/data/met-office/open-license/shapefiles/USA/US_COUNTY_POP.shp'
    JSON_READ = './shapefile_attributes.json'
    CSV_FOLDER = '/data/share/shapecutting/US_COUNTY_POP/'
    CSV_NAME = 'metoffice_global_daily_mask'
    MASK = False
```

### Run with user specified arguments at runtime
```
$ python shapecutter_script.py 20200601 20200630 -s /path/to/shapefile -o /path/to/output/folder
```
This will run the script for the daterange 01-June-2020 to 30-June-2020, with shapefile at `/path/to/shapefile` and save the resulting CSV files in `/path/to/output/folder`. All other parameters use default values.

Here is a list of all the possible parameters you can specify at runtime
```
positional arguments:   
  dt_start                      Start date as YYYYMMDD
  dt_end             (Optional) End date as YYYYMMDD

optional arguments:
  -h, --help                    show this help message and exit
  -d, --data PATH               Path to gridded data
      --skip FNAME [FNAME ...]  Filename(s) to skip
      --filename-format FNAME   Gridded data filename format
  -s, --shapefile PATH          Path to shapefile
  -j, --json PATH               Path to shapefile attributes JSON
  -o, --output PATH             Path to folder to write output CSV files
      --csv-name FNAME          Filename of output CSV files
  -m, --mask [BOOL]             Use masking for gridcell overlap

```


## Software Environment
You an use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to install the right software dependencies, including Jupyter, as specified in the `environment.yaml`:

```
$ conda env create --file environment.yaml
```

### Making conda environment available to Jupyter Notebooks
You can make your conda environment available as a Jupyter Notebook kernel by doing the following:
```
$ conda activate shapecutter
$ python -m ipykernel install --name "shapecutter" --display-name "Python (shapecutter)" --user
```
