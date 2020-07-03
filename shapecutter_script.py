#!/root/my-conda-envs/covid-19/bin/python
# coding: utf-8

## Setup
#Data
import iris
import cartopy.io.shapereader as shpreader
import pandas as pd
import geopandas
import cftime
import datetime

#System
import os
import sys
import glob
import json
import shutil
import argparse

#Met Office utils
import shape_utils as shape

#Supress warnings
import warnings
warnings.filterwarnings('ignore')


## Functions

def get_shapefile_extent(shapefile, buffer=0, **kwargs):
    '''
    Return (x0, x1, y0, y1) extent of total shapefile geometries
    '''
    #Load shapefile into geopandas dataframe
    gdf = geopandas.read_file(shapefile, **kwargs)
    
    #Get total bounds of shapefile
    wsen = gdf.total_bounds
    
    #Rearrange extent from (x1, y1, x2, y2) to (x1, x2, y1, y2), adding buffer
    extent = (wsen[0]-buffer, wsen[2]+buffer, wsen[1]-buffer, wsen[3]+buffer)
    
    return extent

def get_shape_record(target, shape_reader, attribute):
    '''
    Get a record from the shape_reader with a target attribute.
    '''
    result = None
    for record in shape_reader.records():
        shape_id = record.attributes[attribute]
        if shape_id == target:
            result = record
            break
    if result is None:
        emsg = f'Could not find record with {attribute} = "{target}".'
        raise ValueError(emsg)
    return result

def get_cube_extent(cube, buffer=0):
    '''
    Return (x0, x1, y0, y1) extent of cube's spatial coordinates
    with optional buffer
    '''
    extent = (cube.coord(axis='x').points[0]  - buffer,
              cube.coord(axis='x').points[-1] + buffer,
              cube.coord(axis='y').points[0]  - buffer,
              cube.coord(axis='y').points[-1] + buffer,
             )
    return extent

def get_cell_method(cube, coord='time', exclude_interval='1 hour'):
    '''
    Get the cell method with coord in coord_names
    '''
    result = None
    for method in cube.cell_methods:
        if coord in method.coord_names and exclude_interval not in method.intervals:
            result = method.method
            break
    
    return result

def parse_data_name(cube):
    '''
    Parse the name, cell methods and units in a cube to return a column name
    To be used in a Pandas DataFrame
    '''
    name = cube.name()
    time_method = get_cell_method(cube, 'time').replace('imum', '')
    space_method = get_cell_method(cube, 'longitude')
    units = cube.units
    
    if name == 'm01s01i202':
        name = 'short_wave_radiation'
        if space_method.startswith('var'):
            units = 'W2 m-4'
        else:
            units = 'W m-2'
    
    if space_method:
        result = f'{name}_{time_method}_{space_method.replace("imum", "")} ({units})'
    else:
        result = f'{name}_{time_method} ({units})'
    
    return result

def get_date(dt):
    '''
    Return date from datetime-like object dt
    '''
    return datetime.date(dt.year, dt.month, dt.day)

def get_column_order(start, end):
    '''
    Combine start and end, with end in alphanumerical order
    Returns a tuple of their combination
    '''
    starts = tuple(start)
    
    ends = tuple(sorted([c for c in end if c not in starts]))
    
    return starts+ends

def extract_collapse_df(shape_id, cubes, mask=False, **kwargs):
    '''
    Extract subcubes from cubes using geometry of shape_id
    Collapse the cube acros x and y coords to get the MEAN and VARIANCE
    Collect data in a dataframe for this shape_id
    
    Extract method:
        Extracts XY bounding box around geometry
        Refer to [Method](#Method) at top of this notebook for detailed description
        
    Arguments:
        shape_id (str): ID of geometry used for subsetting
        cubes (iris.CubeList): List of Iris cubes to be subsetted
        
    Returns: 
        df (pandas.DataFrame): DataFrame containing shape_id attributes 
                               and MEAN+VARIANCE of data in cubes for shape_id geometry
    '''
    #Create a Shape object from the record for shape_id
    region = get_shape_record(shape_id, **kwargs)
    shp = shape.Shape(region.geometry, region.attributes)
    
    #Extract sub_cubes from cubes using shp
    sub_cubes = shp.extract_subcubes(cubes, mask=mask)
    
    #Collapse cubes across x and y coords with to get mean and variance
    mean_cubes = [cube.collapsed([cube.coord(axis='x'),cube.coord(axis='y')], iris.analysis.MEAN) for cube in sub_cubes]
    var_cubes = [cube.collapsed([cube.coord(axis='x'),cube.coord(axis='y')], iris.analysis.VARIANCE) for cube in sub_cubes]
    
    #Line up data and column names for Pandas DataFrame
    time = mean_cubes[0].coord('time')
    length = len(time.points)
    data = {'shapefile': [SHAPEFILE_NAME]*length}
    data.update({name: [region.attributes[name]]*length for name in SHAPE_ATTRS})
    data.update({'date': [get_date(cell.point) for cell in time.cells()]})
    data.update({parse_data_name(cube): cube.data for cube in mean_cubes})
    data.update({parse_data_name(cube): cube.data for cube in var_cubes})
    
    #Get a column order so that all dataframes have the same column order
    column_order = get_column_order(['shapefile']+list(SHAPE_ATTRS)+['date'], end=data.keys())
    
    #Create DataFrame
    df = pd.DataFrame(data, columns=column_order)

    return df

def get_csv_name(folder, shapefile, shape_id, data_name, start_dt, end_dt, separator='_', dtfmt='%Y%m%d'):
    '''
    Parse inputs to create the name for a CSV file
    
    Arguments:
        folder (str): Directory to write files
        shapefile (str): Name of shapefile used
        shape_id (str): ID of the shape geometry
        data_name (str): Additional name to add to the filename
        start_dt (datetime): Datetime object denoting start of data validity
        end_dt (datetime): Datetime object denoting end of data validity
        separator (str): Seperator used in CSV (Default = '_')
        dtfmt (str): Format string for datetimes (Default = '%Y%m%d')
    '''
    extension = '.csv'
    
    #Cut out underscores from shapefile name to reduce overall length
    shapefile = shapefile.replace('_', '')
    
    #Format the daterange that be at the end of the filename
    if start_dt==end_dt:
        dt_range = f"{start_dt.strftime(dtfmt)}"
    else:
        dt_range = f"{start_dt.strftime(dtfmt)}-{end_dt.strftime(dtfmt)}"
    
    #Join all the filename parts with separator
    if shape_id:
        shape_id = str(shape_id).replace('_', '-').replace('.', '-')
        filename = separator.join([shapefile, shape_id, data_name, dt_range])
    else:
        filename = separator.join([shapefile, data_name, dt_range])
    
    #Join filename with folder
    filepath = os.path.join(folder, f"{filename}{extension}")
    
    return filepath

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

'''--------------MAIN SCRIPT--------------'''

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
    
    # Start timer
    import time
    t0 = time.time()
    print(f'Shapecutting script started')

    # Parse commandline arguments
    print('\n---ARGS---')
    parser = argparse.ArgumentParser(prog='Shapecutter', description='Cut out average values from gridded data')

    parser.add_argument('dt_start', type=str, help='Start date as YYYYMMDD')
    parser.add_argument('dt_end', type=str, nargs='?', default=None, help='End date as YYYYMMDD')

    parser.add_argument('-d', '--data', nargs=1, default=[DATA_FOLDER], type=str, metavar='PATH', help='Path to gridded data')
    parser.add_argument('--skip', nargs='+', default=SKIP_FILES, type=str, metavar='FNAME', help='Filename(s) to skip')
    parser.add_argument('--filename-format', nargs=1, default=[FILENAME_FORMAT], type=str, metavar='FNAME', help='Gridded data filename format')
    parser.add_argument('-s', '--shapefile', nargs=1, default=[SHAPEFILE], type=str, metavar='PATH', help='Path to shapefile')
    parser.add_argument('-j', '--json', nargs=1, default=[JSON_READ], type=str, metavar='PATH', help='Path to shapefile attributes JSON')
    parser.add_argument('-o', '--output', nargs=1, default=[CSV_FOLDER], type=str, metavar='PATH', help='Path to folder to write output CSV files')
    parser.add_argument('--csv-name', nargs=1, default=[CSV_NAME], type=str, metavar='FNAME', help='Filename of output CSV files')
    parser.add_argument('-m', '--mask', nargs='?', const=True, default=MASK, type=bool, metavar='BOOL', help=f'Use masking for gridcell overlap (Default={MASK})')
    
    args = parser.parse_args()
    print(f'Args:\n{args}')

    # Replace default with commandline args
    dt_start = args.dt_start
    dt_end = dt_start if args.dt_end==None else args.dt_end
    DATA_FOLDER = args.data[0]
    SKIP_FILES = args.skip
    FILENAME_FORMAT = args.filename_format[0]
    SHAPEFILE = args.shapefile[0]
    JSON_READ = args.json[0]
    CSV_FOLDER = args.output[0]
    CSV_NAME = args.csv_name[0]
    MASK = args.mask

    # Calculate daterange
    print('\n---DATERANGE---')
    start = datetime.datetime.strptime(dt_start, "%Y%m%d").date()
    stop = datetime.datetime.strptime(dt_end, "%Y%m%d").date()
    step = datetime.timedelta(days=1)
    DATERANGE = pd.date_range(start, stop, freq=step)
    print(f'Start date: {dt_start}')
    print(f'End date:   {dt_end}')
    print(f'Daterange:  {DATERANGE}')

    # Data vars
    print('\n---DATA---')
    FILENAMES = list(DATERANGE.strftime(FILENAME_FORMAT))
    print(f'Data folder: {DATA_FOLDER}')
    print(f'Filenames:   {FILENAMES}')

    # Shapefile vars
    print('\n---SHAPEFILES---')
    SHAPEFILE_NAME = SHAPEFILE.split('/')[-1].split('.')[-2]
    print(f'Shapefile name:   {SHAPEFILE_NAME}')

    # Shapefile attributes
    with open(JSON_READ) as file:
        SHAPE_ATTRS = tuple(json.load(file)['shapefile_attributes'][SHAPEFILE_NAME])
    print(f'Shape attributes: {SHAPE_ATTRS}')

    # CSV vars
    print('\n---CSVs---')
    if dt_start == dt_end:
        TEMP_CSV_FOLDER = os.path.join(CSV_FOLDER, f'temp_csvs_{dt_start}')
    else:
        TEMP_CSV_FOLDER = os.path.join(CSV_FOLDER, f"temp_csvs_{dt_start}-{dt_end}")
    print(f'Temporary CSV folder: {TEMP_CSV_FOLDER}')
    CSV_FINAL = get_csv_name(CSV_FOLDER, SHAPEFILE_NAME, None, CSV_NAME, DATERANGE[0], DATERANGE[-1])
    print(f'Collated CSV file:    {CSV_FINAL}')

    # Get all data filepaths
    print('\n---PROCESSING---')
    FILEPATHS = {}
    for path in os.listdir(DATA_FOLDER):
        if path not in SKIP_FILES:
            FILEPATHS[path] = []
            for filename in FILENAMES:
                FILEPATHS[path].extend(glob.glob(os.path.join(DATA_FOLDER, path, filename)))
    # print(FILEPATHS)

    # Initiate shape_reader
    SHAPE_READER = shpreader.Reader(SHAPEFILE)
    SHAPE_EXTENT = get_shapefile_extent(SHAPEFILE, buffer=1)
    SHAPE_IDS = tuple(record.attributes[SHAPE_ATTRS[0]] for record in SHAPE_READER.records())
    # print(SHAPE_EXTENT)

    # Load cubes
    cubes = iris.cube.CubeList([])
    # for var in ['t1o5m_mean']:
    # for var in ['sh_mean', 't1o5m_mean']:
    # for var in ['sh_mean', 'sw_mean', 't1o5m_mean']:
    # for var in ['sh_mean', 'sw_mean', 'precip_mean', 't1o5m_mean']:
    for var in FILEPATHS.keys():
        cubes.extend(iris.load(FILEPATHS[var]))
    # print(cubes)

    # Subset cubes
    x_axis = cubes[0].coord(axis='x')
    y_axis = cubes[0].coord(axis='y')
    x_extent = iris.coords.CoordExtent(x_axis, SHAPE_EXTENT[0], SHAPE_EXTENT[1])
    y_extent = iris.coords.CoordExtent(y_axis, SHAPE_EXTENT[2], SHAPE_EXTENT[3])
    SHAPE_CUBES = iris.cube.CubeList([cube.intersection(x_extent, y_extent) for cube in cubes])
    # print(SHAPE_CUBES)

    # Create temporary CSV folder
    if not os.path.exists(TEMP_CSV_FOLDER):
        os.makedirs(TEMP_CSV_FOLDER)

    # Loop through all regions, saving out temporary CSV for each one
    start = 0
    stop = len(SHAPE_IDS)
    for n, shape_id in enumerate(SHAPE_IDS[start:stop]):
        try:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            percent = f'{100*(n+start)/stop:.1f}%'
            fname = get_csv_name(TEMP_CSV_FOLDER, SHAPEFILE_NAME, shape_id, CSV_NAME, DATERANGE[0], DATERANGE[-1])
            df = extract_collapse_df(shape_id, SHAPE_CUBES, shape_reader=SHAPE_READER, attribute=SHAPE_ATTRS[0], mask=MASK)
            df.to_csv(fname, index=False)
            print(f'  {percent} {now} [{shape_id}] {fname}: Success')
        except Exception as e:
            print(f'x xx.x% {now} [{shape_id}] {fname}: Error \n  x  {e}')

    # Collate temporary CSVs into final CSV
    print('\n---COLLATING---')
    CSVS_READ = glob.glob(os.path.join(TEMP_CSV_FOLDER, '*.csv'))
    df = pd.concat([pd.read_csv(csv) for csv in CSVS_READ], ignore_index=True)
    df.sort_values(by=SHAPE_ATTRS[0]).to_csv(CSV_FINAL, index=False)
    print(f'Final CSV file written to:    {CSV_FINAL}')

    # Delete temporary CSVs folder
    print('\n---TIDYING---')
    if os.path.exists(CSV_FINAL):
        try:
            shutil.rmtree(TEMP_CSV_FOLDER)
            print(f'Temporary CSV folder removed: {TEMP_CSV_FOLDER}')
        except Exception as e:
            print(f'ERROR: {e}')

    # Print total time for pipeline
    print('\n---FINISHING---')
    t1 = time.time()
    total_time = datetime.timedelta(seconds=(t1-t0))
    print(f"Finished\nTotal time: {total_time}")

