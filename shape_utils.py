"""Copyright 2018 British Crown Copyright, Met Office

ASCEND is a free software developed within the Met
Office. ASCEND is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY, without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. It is released under the BSD-3-Clause license.

"""

import sys
import os
import warnings
import importlib  # convenience wrappers for __import__()
import copy       # shallow and deep copy operations
import glob

import shapely
import shapely.ops
import shapely.geometry as sgeom
import iris
import iris.plot as iplt
from iris.cube import Cube, CubeList
import iris.coord_systems as ics
import iris.analysis.geometry as iag
import numpy as np
import cartopy
from cartopy.feature import NaturalEarthFeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
import shapefile
import scipy.cluster.hierarchy as hcluster


# ------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ------------------------------------------------------------------------------

VERBOSE = False
DEFAULT_CS = ics.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
CUBE_ASCSHADE_KEY = 'AscShapeAtt'
LONGNAME_WEIGHT = 'weights_of_intersection'
LONGNAME_INTERSECT = 'intersection'
SHAPEFILE_TYPES = {'point': (shapefile.POINT, shapefile.Writer.point),
                   'line': (shapefile.POLYLINE, shapefile.Writer.line),
                   'polygon': (shapefile.POLYGON, shapefile.Writer.poly)}
SHAPEFILE_FIELD_TYPES = {int: {'fieldType': 'N', 'decimal': 0},
                         float: {'fieldType': 'N', 'decimal': 1},
                         str: {'fieldType': 'C', 'decimal': 0}}


# ------------------------------------------------------------------------------
# CLASSES
# ------------------------------------------------------------------------------

class ShapeList(list):
    """A list class holding multiple :class:`.Shape` objects.

    Arguments:
        list_of_shapes (list): A list of :class:`.Shape` objects
    """
    def __add__(self, other):
        return ShapeList(list.__add__(self, other))

    def __new__(cls, list_of_shapes=None):
        """Given a list of :class:`.Shape` objects, return a
        :class:`.ShapeList` instance."""
        shapelist = list.__new__(cls, list_of_shapes)
        if not all([isinstance(a_shape, Shape) for a_shape in shapelist]):
            raise ValueError('All items in list must be Shape objects.')
        return shapelist

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        result = ['{}: {}'.format(i, a_shape.summary(short=True))
                  for i, a_shape in enumerate(self)]
        if result:
            result = '\n'.join(result)
        else:
            result = '< No Shapes >'
        return result

    def buffer(self, distance, **kwargs):
        """Perform :py:mod:`shapely` buffer function on all shapes.

        Arguments:
            distance (float): The distance in shape units (normally degrees)
                              to be buffered

        Returns:
            A new :class:`ShapeList` of buffered shapes

        See also:
            :meth:`shapely.geometry.base.BaseGeometry.buffer` for details of
            keyword arguments.
        """
        buffer_list = [a_shape.buffer(distance, **kwargs) for a_shape in self]
        return ShapeList(buffer_list)

    def cascaded_union(self):
        """Perform :py:mod:`shapely` cascaded union on all shapes.

        Returns:
            A new :class:`.Shape` object
        """
        return manipulate(self, 'shapely.ops.cascaded_union')

    def filter(self, **kwargs):
        """Filter shapes in the :class:`.ShapeList` object based on the given
        keywords.

        Arguments:
            kwargs: Set keyword arguments for filtering (see below)

        Returns:
            A new :class:`.ShapeList` object of matching shapes

        Note:
            Only returns objects which include the filter key (case insensitive)
            and values, i.e. if you searched for continents but the objects did
            not have it as an attribute then no objects would be returned.

        Examples:
            Return all objects with attribute *Continent* = *Europe*:

            a_shapelist.filter(Continent='Europe')

            or

            filters = {'Continent': 'Europe'}
            a_shapelist.filter(**filters)

            Return objects with attribute *Continent* = *Europe* or *Africa*:

            a_shapelist.filter(Continent=['Europe', 'Africa'])

            Return objects with attribute *Continent* = *Europe* or *Africa*
            and which have *lastcensus* = 1979.0:

            a_shapelist.filter(Continent=['Europe', 'Africa'], lastcensus=1979.0)
        """
        filtered_list = ShapeList()
        if not kwargs:
            return self

        # Loop over all filters provided, if the object has the attribute
        # and its key matches the provided filter value then store True,
        # else store match (search is case insensitive)

        for a_shape in self:
            keep = []  # will be boolean list of matches to filters
            shape_attributes = {a.lower(): v  # copy of attributed but with lower case key
                                for a, v in a_shape.attributes.items()}
            for key, value in kwargs.items():  # loop over requested filter options
                filter_key = key.lower()

                # set filter value as a list
                filter_value = value if isinstance(value, (list, tuple)) else [value]

                # if filter key is not in the shape then do not keep it
                if filter_key not in shape_attributes.keys():
                    keep.append(False)
                    continue

                # check if the shapes value is within the filter value
                if shape_attributes[filter_key] in filter_value:
                    keep.append(True)
                else:
                    keep.append(False)

            # only keep shapes which pass ALL filter options
            if all(keep):
                filtered_list.append(a_shape)

        return filtered_list

    def plot(self, **kwargs):
        """Show the shape on a map.

        See also:
            :func:`plot` for details of arguments.
        """
        return plot(self, **kwargs)

    def remove(self, **kwargs):
        """Remove shapes from the :class:`.ShapeList` object according to the
        keyword arguments which should match attributes of the shapes in the
        :class:`.ShapeList` object.

        Arguments:
            kwargs: Attributes to use to identify shapes to be removed (these
                    attributes should match those of the shapes)

        Returns:
            A copy of the :class:`.ShapeList` object with all matches to the
            keyword arguments removed
        """
        return ShapeList(set(self).difference(set(self.filter(**kwargs))))

    def save_shp(self, target, filetype, overwrite=False):
        """Save the :class:`.ShapeList` object to a shapefile.

        See also:
              :func:`save_shp` for details of arguments.
        """
        save_shp(self, target, filetype, overwrite)

    def show(self, **kwargs):
        """Show the shapes on a map.

        See also:
            :func:`show` for details of arguments.
        """
        show(self, **kwargs)

    def unary_union(self):
        """Perform :py:mod:`shapely` unary union on all shapes.

        Returns:
            A new :class:`.Shape` object
        """
        return manipulate(self, 'shapely.ops.unary_union')


class Shape(object):
    """A class for holding geometry information.

    Arguments:
        data        : A :class:`shapely.geometry` instance
        attributes  : A dictionary of metadata for the new object
        coord_system: The :class:`iris.coord_systems.CoordSystem` coordinate
                      system of the object if known (defaults to a regular
                      latitude/longitude system otherwise)

    Attributes:
        is_valid (bool)  : Whether it is a valid :py:mod:`shapely.geometry`
        data             : A :class:`shapely.geometry` instance
        attributes (dict): The object metadata
        coord_system     : The shape's coordinate system
    """
    def __init__(self, data, attributes, coord_system=DEFAULT_CS):
        """Create a new class instance."""
        if not isinstance(attributes, dict):
            msg = ('Initialization; invalid argument type, '
                   'dictionary required, not {}')
            raise TypeError(msg.format(type(attributes)))

        try:
            check_geometry_validity(data)
            valid = True
        except(TypeError, shapely.geos.TopologicalError):
            valid = False

        if not isinstance(coord_system, iris.coord_systems.CoordSystem):
            msg = ('Initialization; invalid argument type, '
                   'Iris coord_system required, not {}')
            raise TypeError(msg.format(type(coord_system)))

        self.is_valid = valid
        self.data = data
        self.attributes = attributes
        self.coord_system = coord_system

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return self.summary()

    def append_attributes(self, attr_dict):
        """Append values of the given attribute dictionary/dictionaries to the
        corresponding values of the object's :attr:`.attributes` dictionary
        (separation character is '|').

        Arguments:
            attr_dict (list or dict): A dictionary or a list of dictionaries of
                                      :class:`.Shape`-like attributes
        """
        if isinstance(attr_dict, dict):
            dict_list = [attr_dict]
        elif isinstance(attr_dict, list):
            dict_list = attr_dict
        else:
            raise TypeError('`attr_dict` must be a dictionary or a list of '
                            'dictionaries of Shape attributes')
        for a_dict in dict_list:
            for key, value in a_dict.items():
                if key in self.attributes.keys():
                    x = self.attributes[key]
                    self.attributes[key] = '{} | {}'.format(x, value)
                else:
                    self.attributes[key] = value
        return

    def buffer(self, distance, **kwargs):
        """Add a buffer to the shape.

        Arguments:
            distance (float): The distance in shape units (normally degrees) to
                              to be buffered

        Returns:
            The shape expanded by `distance`

        See also:
            :meth:`shapely.geometry.base.BaseGeometry.buffer` for details of
            keyword arguments.
        """
        return manipulate(self, 'buffer', distance, **kwargs)

    def cluster_shapes(self):
        """Return a list of clustered shapes."""
        cluster_types = (sgeom.MultiPoint, sgeom.MultiPolygon)
        if isinstance(self.data, cluster_types) and self.sparse():
            clustered_geoms = cluster_geometries(self.data)
            clustered_shapes = [Shape(geom, self.attributes, self.coord_system)
                                for geom in clustered_geoms]
        else:
            clustered_shapes = [self]
        return clustered_shapes

    def compare_to_cube_attributes(self, cube, key, default=True):
        """Compare the shape attributes against a cube's attribute key.

        Arguments:
            cube          : An :class:`iris.cube.Cube` object
            key     (str) : The cube attributes dictionary key to be checked
            default (bool): The return value should the cube not have the given
                            attribute key

        Returns:
            `True` if the shape and the cube share the attribute, `False`
            otherwise

        """
        smatch = default
        check_cube_instance(cube)
        if key in cube.attributes:
            smatch = self.attributes == cube.attributes[key]
        return smatch

    def constrain_cube(self, cube, border=0.0):
        """Constrain the given Iris cube to the extent of the shape.

        Arguments:
            cube          : The :class:`iris.cube.Cube` object to constrain
            border (float): Additional border in shape units (generally
                            degrees) to add to the boundaries

        Returns:
            A cube constrained to the extent of the shape or `None` if no data
            found
        """
        check_cube_instance(cube)
        cube_xy_guessbounds(cube)
        cube = cube.copy()

        if not self.coord_system == get_cube_coord_system(cube):
            if VERBOSE:
                msg = "Constrain cube: {} does not equal {}"
                print(msg.format(self.coord_system,
                                 get_cube_coord_system(cube)))
            return None

        latitude, longitude = cube_primary_xy_coord_names(cube)
        geometry_bounds = np.array(self.data.bounds)
        bound_index = np.array([True, True, False, False])
        geometry_bounds[bound_index] -= border
        geometry_bounds[~bound_index] += border

        if self.constrain_cube_full_grid_request(cube, border=border):
            self.x_rotate_cube(cube)
            longitude = None  # do not perform constrain on longitude

        bound_cube = constrain_cube_to_bounds(cube, geometry_bounds,
                                              latitude, longitude)
        return bound_cube

    def constrain_cube_full_grid_request(self, cube, border=0.0):
        """Check if the shape's bounds are equal to the full cube longitude grid
        (and this is the entire earth).

        Arguments:
            cube          : The :class:`iris.cube.Cube` object to be tested
                            against the geometry's bounds
            border (float): Additional border in shape units (generally
                            degrees) to add to the boundaries

        Returns:
            `True` if full longitude grid is needed for constraining, `False`
            otherwise

            This method performs the following tests:

            1. Is the requested shape size greater than or equal to the
               longitude modulus?
            2. Is the difference between the grid length and geometry less
               than one longitude step?
            3. Is the grid distance equal to the longitude modulus?
            4. Is the bound distance equal to the longitude modulus?

            The returned result is: `test_1 or (test_2 and test_3) or
            (test_2 and test_4)`
        """
        check_cube_instance(cube)
        latitude, longitude = cube_primary_xy_coord_names(cube)
        geometry_bounds = np.array(self.data.bounds)
        x_dist = geometry_bounds[2] - geometry_bounds[0]
        xcord = cube.coord(longitude)
        total_x = xcord.points[-1] - xcord.points[0]
        dx = xcord.points[1] - xcord.points[0] if xcord.points.size > 1 else 0.
        tol = 1e-12

        if xcord.units.modulus is None:
            if VERBOSE:
                msg = '{} modulus is None, please check results carefully'
                print(msg.format(longitude))
            return False

        test_geom_xmod = x_dist + (2.0 * border) >= (xcord.units.modulus - tol)
        test_grid_geom = np.abs(x_dist - total_x) <= np.abs(dx + tol)
        test_grid_xmod = np.abs(total_x - xcord.units.modulus) < tol
        test_bond_xmod = False

        if xcord.has_bounds():
            bound_x = np.abs(xcord.bounds[0, 0] - xcord.bounds[-1, 1])
            test_bond_xmod = bound_x >= (xcord.units.modulus - tol)

        if (test_geom_xmod or
           (test_grid_geom and test_grid_xmod) or
           (test_grid_geom and test_bond_xmod)):
            return True

        return False

    def copy(self, new_geom=None, new_attr=None, new_cs=None):
        """Returns a new :class:`.Shape` object of this shape.

        Arguments:
            new_geom       : New :class:`shapely.geometry.base.BaseGeometry`
                             object for the copied shape (i.e. the object's
                             :attr:`.data` attribute)
            new_attr (dict): New attributes for the copied shape (i.e. the
                             object's :attr:`.attributes` attribute)
            new_cs         : New :class:`iris.coord_systems.CoordSystem`
                             coordinate system for the copied shape (i.e. the
                             object's :attr:`.coord_system` attribute)

        Returns:
            A copy of the :class:`.Shape` object

        Note:
            This method does not transform any data (e.g. changing the
            coordinate system does not update the actual geometry).
        """
        shape_copy = copy.deepcopy(self)
        data = shape_copy.data if new_geom is None else new_geom
        attr = shape_copy.attributes if new_attr is None else new_attr
        csys = shape_copy.coord_system if new_cs is None else new_cs
        return Shape(data, attr, coord_system=csys)

    def cube_2d_weights(self, cube, intersection=True):
        """Get the bidimensional (latitude/longitude) weights of intersection
        between the shape and a cube.

        Arguments:
            cube               : An :class:`iris.cube.Cube`) object (can be
                                 multidimensional)
            intersection (bool): Whether to use faster methodology to only get
                                 the points which intersect AT ALL with the
                                 geometry

        Returns:
            A cube containing the shape intersection weights

        Note:
            Only the primary latitude and longitude coordinates are preserved.

        """
        check_cube_instance(cube)
        cube_xy_guessbounds(cube)
        xy_coordinates = cube_primary_xy_coord_names(cube)
        try:
            cube.data.shape  # make sure data has been loaded
        except Exception as e:
            raise e
        first_slice = cube.copy().slices(xy_coordinates).next()
        remove_non_lat_lon_cube_coords(first_slice, xy_coordinates)

        # generate the weights

        if intersection:
            long_name = 'intersection'
            normalize = False
        else:
            long_name = 'weights_of_intersection'
            normalize = True

        weights = self.cube_intersection_mask(first_slice,
                                              intersection,
                                              normalize)

        # create the weights cube

        weights_cube = iris.cube.Cube(weights,
                                      standard_name=None,
                                      long_name=long_name,
                                      units='1')

        for coord_index, coord in enumerate(xy_coordinates):
            weights_cube.add_dim_coord(first_slice.coord(coord), coord_index)

        # add attribute information if requested

        weights_cube.attributes[CUBE_ASCSHADE_KEY] = self.attributes
        return weights_cube

    def cube_intersection_mask(self, cube, intersect=True, normalize=False):
        """Get the mask cube of the intersection between the given shape and
        cube.

        Arguments:
            cube            : An :class:`iris.cube.Cube` slice
                              with x and y coordinates
            intersect (bool): `True` to apply a binary intersection method,
                              `False` to compute area overlap
            normalize (bool): Whether to calculate each individual cell weight
                              as the cell area overlap between the cell and the
                              given shape divided by the total cell area

        Returns:
            A cube containing the masked data
        """
        check_cube_instance(cube)
        check_2d_latlon_cube(cube)
        y_name, x_name = cube_primary_xy_coord_names(cube)
        coord_dims = [cube.coord_dims(x_name), cube.coord_dims(y_name)]
        xmod = cube.coord(x_name).units.modulus
        ymod = cube.coord(y_name).units.modulus
        cube_of_zeros = zero_cube(cube)

        # keep track of progress
        total_progress = 0.0
        percent_count = 1
        msg = ("Shape analysis for " + cube.name())

        # intersection of cube bounding box and shape
        shp_intersect = cube_bbox_shape_intersection(cube, self)
        clustered_shapes = shp_intersect.cluster_shapes()

        # loop through each cluster
        for shp in clustered_shapes:
            if not shp.data.bounds:
                continue
            # constrain the cube to the geometry's bounds
            geo_cube = shp.constrain_cube(cube)
            if not geo_cube:
                continue

            # workaround for dim coord to scalar coord
            # i.e. make sure geo_cube is 2d
            for idx, coord in enumerate([x_name, y_name]):
                if geo_cube.coord_dims(coord) != coord_dims[idx]:
                    geo_cube = iris.util.as_compatible_shape(geo_cube, cube)
            zero_cube(geo_cube, inplace=True)

            # perform the masking
            tpoints = np.float(geo_cube.data.size)
            shp_area_factor = geometry_factor(shp_intersect.data, shp.data)

            if VERBOSE:
                progress = 0.0  # initialise outside of loop in case it's empty

            for count, idx in enumerate(np.ndindex(geo_cube.shape)):
                if VERBOSE:
                    progress = 100.0 * count * shp_area_factor / tpoints
                    if total_progress + progress >= percent_count:
                        update_progress(percent_count, msg, sys.stdout)
                        percent_count += 1

                # get the bounds of the grid cell
                xi = idx[geo_cube.coord_dims(x_name)[0]]
                yi = idx[geo_cube.coord_dims(y_name)[0]]
                x0, x1 = geo_cube.coord(x_name).bounds[xi]
                y0, y1 = geo_cube.coord(y_name).bounds[yi]

                # Iris 1.8 workaround; re-base values to modulus
                if xmod:
                    x0, x1 = rebase_values_to_modulus((x0, x1), xmod)
                if ymod:
                    y0, y1 = rebase_values_to_modulus((y0, y1), ymod)

                # create a new polygon of the grid cell and check intersection
                poly = sgeom.box(x0, y0, x1, y1)
                intersect_bool = poly.intersects(shp.data)
                if intersect_bool:
                    if intersect:
                        geo_cube.data[idx] = intersect_bool
                    else:
                        geo_cube.data[idx] = poly.intersection(shp.data).area
                        if normalize:
                            geo_cube.data[idx] /= poly.area

            # put constrained cube back into main cube
            rollx = shp.constrain_cube_full_grid_request(geo_cube, 0.0)
            rcube = reintersect_2d_slice(zero_cube(cube),
                                         geo_cube, roll_lon=rollx)

            iris.analysis.maths.add(cube_of_zeros, rcube, in_place=True)
            if VERBOSE:
                total_progress += progress

        if VERBOSE:
            update_progress(100, msg, sys.stdout)

        return cube_of_zeros.data

    def difference(self, a_shape):
        """Perform :py:mod:`shapely` difference with another :class:`.Shape`
        object.

        Arguments:
            a_shape (:class:`.Shape`): A shape

        Returns:
            The difference between both :class:`.Shape` objects
        """
        return manipulate(self, 'difference', a_shape)

    def extract_subcube(self, cube, border=0.0, mask=False, **kargs):
        """Extract cube to the extent of the shape.

        Arguments:
            cube          : The :class:`iris.cube.Cube` object to constrain
            border (float): Additional border in shape units
                            to add to the boundaries in all directions
            mask          : True/False - mask the subcube data using the shape
            **kargs       : keyword arguments for cube masking
                            (e.g. minimum_weight)

        Returns:
            An Iris cube constrained to the shape

        Note:
            For data using a rotated pole grid, the longitude coordinate may
            be returned modulo 360 degrees.
        """
        check_cube_instance(cube)
        scube = self.extract_subcubes([cube], border=border)[0]
        if mask:
            self.mask_cube_inplace(scube, **kargs)
        return scube

    def extract_subcubes(self, cubes, border=0.0, mask=False, **kargs):
        """Extract cubes to the extent of the shape.

        Arguments:
            cubes         : A list of :class:`iris.cube.Cube` objects to
                            constrain
            border (float): Additional border in shape units
                            to add to the boundaries in all directions
            mask          : True/False - mask the subcube data using the shape
            **kargs       : keyword arguments for cube masking
                            (e.g. minimum_weight)
        Returns:
            A list of cubes (:class:`iris.cube.CubeList`) constrained to the
            extent of the shape

        Note:
            For data using a rotated pole grid, the longitude coordinate may
            be returned modulo 360 degrees.
        """
        if not isinstance(cubes, (CubeList, list, tuple)):
            msg = "Expected iterable object (cubelist, list, tuple), got {}"
            raise TypeError(msg.format(type(cubes)))
        subcubes = CubeList()
        for cube in cubes:
            check_cube_instance(cube)
            rgeometry = self.transform_coord_system(cube)
            border_cube = rgeometry.constrain_cube(cube, border)
            subcubes.append(border_cube)
        if mask:
            self.mask_cubes_inplace(subcubes, **kargs)
        return subcubes

    def find_weights_cube(self, cube, weight_cubes):
        """Find the matching weights cube for a given cube (checks coordinates
        and attributes).

        Arguments:
            cube        : An :class:`iris.cube.Cube` object
            weight_cubes: A list or :class:`iris.cube.CubeList` object of
                          weights cubes

        Returns:
            The matching cube if `weights_cubes` contains a cube matching
            `cube`, `None` otherwise
        """
        for weight_cube in weight_cubes:
            if not weight_cube:
                continue
            check_cube_instance(cube)
            match_grid = compare_iris_cube_grids(cube, weight_cube)
            match_attr = self.compare_to_cube_attributes(weight_cube,
                                                         CUBE_ASCSHADE_KEY)
            if all([match_attr, match_grid]):
                return weight_cube
        return None

    def get_coordinates_as_list(self):
        """Return the shape's coordinates as a list.

        Returns:
            A list of xy coordinates
        """
        shapes = self.data if hasattr(self.data, 'geoms') else [self.data]
        coordinates = []
        for a_shape in shapes:
            ecoords = []
            icoords = []
            try:
                ecoords = list(a_shape.coords)
                icoords = None
            except Exception as e:  # TODO: precise exception (used to be "except:")
                try:
                    ecoords = list(a_shape.exterior.coords)
                    for inner in a_shape.interiors:
                        icoords.append(list(inner.coords))
                except Exception as e:  # TODO: precise exception (used to be "except:")
                    msg = ("Could not extract coordinates "
                           "from geometry type {!r}")
                    warnings.warn(msg.format(type(self.data)))
            coordinates.append(ecoords)
            if icoords:
                for inner in icoords:
                    coordinates.append(inner)
        return coordinates

    def intersection(self, a_shape):
        """Perform :py:mod:`shapely` intersection with another :class:`.Shape`
        object.

        Arguments:
            a_shape (:class:`Shape`): A shape

        Returns:
            The intersection between both :class:`.Shape` objects

        """
        return manipulate(self, 'intersection', a_shape)

    def mask_cube_inplace(self, cube, minimum_weight=None, weight_cubes=None):
        """Mask data **in place** within cube using the shape.

        Arguments:
            cube                  : The :class:`iris.cube.Cube` object to mask
            minimum_weight (float): Mask all grid cells with weights less
                                    than this value (value can be 0.0-1.0; if
                                    0.0 or `None`, the intersection method is
                                    used)
            weight_cubes          : List of :class:`iris.cube.Cube` objects or
                                    :class:`iris.cube.CubeList` object (all
                                    cubes must be bidimensional with x and y
                                    coordinates) which provides the masking
                                    weights if already computed
        Warning:
          Operation is performed in place. If this behaviour is not desired,
          please use :meth:`.Shape.mask_cube`.

        """
        check_cube_instance(cube)
        self.mask_cubes_inplace([cube], minimum_weight, weight_cubes)
        return None

    def mask_cubes_inplace(self, cubes, minimum_weight=None,
                           weight_cubes=None):
        """Mask data **in place** within cubes using the shape.

        Arguments:
            cubes                 : List of :class:`iris.cube.Cube` objects or
                                    :class:`iris.cube.CubeList` object to mask
            minimum_weight (float): Mask all grid cells with weights less
                                    than this value (value can be 0.0-1.0; if
                                    0.0 or `None`, the intersection method is
                                    used)
            weight_cubes          : List of :class:`iris.cube.Cube` objects or
                                    :class:`iris.cube.CubeList` object (all
                                    cubes must be bidimensional with x and y
                                    coordinates) which provides the masking
                                    weights if already computed

        Warning:
            Operation is performed in place. If this behaviour is not desired,
            please use :meth:`.Shape.mask_cubes`.
        """
        if not isinstance(cubes, (CubeList, list, tuple)):
            msg = "Expected iterable object (cubelist, list, tuple), got {}"
            raise TypeError(msg.format(type(cubes)))

        if weight_cubes is None:
            weight_cubes = [None]

        if (not minimum_weight or minimum_weight == 0.0 or
                not isinstance(self.data, (sgeom.Polygon,
                                           sgeom.MultiPolygon))):
            isect = True
            minimum_weight = 1e-5  # minimum weight for masking method
        else:
            isect = False
            res = check_intersection_cubestype(weight_cubes)
            if res:
                msg = ("Intersection cubes provided for weights methodology; "
                       "check results carefully!")
                warnings.warn(msg)

        # loop over cubes, transform the geometry,
        # constrain the cube if needed, then select
        # corresponding weights cube (compute if needed)
        # if no suitable weight_cube found in weight_cubes
        # then create a new one

        for cube in cubes:
            check_cube_instance(cube)
            rgeometry = self.transform_coord_system(cube)
            wcube = rgeometry.find_weights_cube(cube, weight_cubes)
            if not wcube:
                wcube = rgeometry.cube_2d_weights(cube, intersection=isect)
                weight_cubes.append(wcube)
            mask_cube_with_minimum_weights(cube, wcube.data, minimum_weight)

        return None

    def mask_cubes(self, cubes, minimum_weight=None, weight_cubes=None):
        """Mask data within cubes using the shape.

        Arguments:
            cubes                 : List of :class:`iris.cube.Cube` objects or
                                    :class:`iris.cube.CubeList` object to mask
            minimum_weight (float): Mask all grid cells with weights less
                                    than this value (value can be 0.0-1.0; if
                                    0.0 or `None`, the intersection method is
                                    used)
            weight_cubes          : List of :class:`iris.cube.Cube` objects or
                                    :class:`iris.cube.CubeList` object (all
                                    cubes must be bidimensional with x and y
                                    coordinates) which provides the masking
                                    weights if already computed

        Returns:
            A copy of `cubes` with all points not within the shape masked.
        """
        icubes = copy.deepcopy(cubes)
        iweights = copy.deepcopy(weight_cubes)
        self.mask_cubes_inplace(icubes, minimum_weight, iweights)
        return icubes

    def mask_cube(self, cube, minimum_weight=None, weight_cubes=None):
        """Mask data within cube using the shape.

        Arguments:
            cube                  : A :class:`iris.cube.Cube` object to mask
            minimum_weight (float): Mask all grid cells with weights less
                                    than this value (value can be 0.0-1.0; if
                                    0.0 or `None`, the intersection method is
                                    used)
            weight_cubes          : List of :class:`iris.cube.Cube` objects or
                                    :class:`iris.cube.CubeList` object (all
                                    cubes must be bidimensional with x and y
                                    coordinates) which provides the masking
                                    weights if already computed

        Returns:
            A copy of `cube` with all points not within the shape masked.
        """
        check_cube_instance(cube)
        icube = copy.deepcopy(cube)
        iweights = copy.deepcopy(weight_cubes)
        self.mask_cube_inplace(icube, minimum_weight, iweights)
        return icube

    def plot(self, **kwargs):
        """Plot the shape on a map.

        See also:
            :func:`plot` for details of arguments.
        """
        return plot([self], **kwargs)

    def save_shp(self, target, filetype, overwrite=False):
        """Save the object to a shapefile.

        See also:
            :func:`save_shp` for details of arguments.
        """
        save_shp(self, target, filetype, overwrite)

    def show(self, **kwargs):
        """Show the shape on a map.

        See also:
            :func:`show` for details of arguments.
        """
        show([self], **kwargs)

    def sparse(self, tol=0.2, buffer_val=1.0):
        """Determines the sparsity of the geometry, within the bounding box
        of the geometry. This is to catch geometries that are widely spread,
        such as France, where the area of the bounding box is very large,
        but the area of the actual geometry is relatively small.

        Arguments:
            tol (float): the tolerance which determines whether a geometry is
                         sparse. 100 * tol is the percentage of the bounding box
                         area covered by the geometry.
            buffer_val (float): the distance in shape units (normally degrees)
                                to be buffered.
        """
        box_area = sgeom.box(*self.data.bounds).area
        # determine the density of the shape within the bounding box.
        if isinstance(self.data, (sgeom.Polygon, sgeom.MultiPolygon)):
            geo_density = self.data.area / box_area
        else:
            try:
                geo_density = self.buffer(buffer_val).data.area / box_area
            except ZeroDivisionError:
                # single point, so bounding box will have no area
                geo_density = 1.0
        return geo_density < tol

    def summary(self, short=False):
        """Print useful object information.

        Arguments:
            short (bool): Short or long version
        """
        atts = ['ascend.Shape object']
        atts.append('data: {}'.format(type(self.data)))
        count = 0
        max_count = 10 if not short else 1
        for key, value in self.attributes.items():
            if count == max_count:
                break
            atts.append('attributes: {}: {}'.format(key, value))
            count += 1
        atts.append('...')
        atts.append('is_valid: {}'.format(self.is_valid))
        atts.append('coord_system: {}'.format(self.coord_system))
        delim = '\n' if not short else ', '
        return delim.join(atts)

    def symmetric_difference(self, a_shape):
        """Perform :py:mod:`shapely` symmetric difference with another
        :class:`.Shape` object.

        Arguments:
            a_shape (:class:`Shape`): A shape

        Returns:
            The difference between both :class:`.Shape` objects
        """
        return manipulate(self, 'symmetric_difference', a_shape)

    def transform_coord_system(self, target):
        """Project the shape onto another coordinate system.

        Arguments:
            target: The target :class:`iris.coord_systems.CoordSystem`
                    or a :class:`iris.cube.Cube` object defining the coordinate
                    system to which the shape should be transformed

        Returns:
            A transformed shape (copy)
        """
        if isinstance(target, Cube):
            check_cube_instance(target)
            cube_xy_guessbounds(target)
            target_cs = get_cube_coord_system(target)
        elif isinstance(target, iris.coord_systems.CoordSystem):
            target_cs = target
        else:
            msg = "Iris cube or coord_system required, got {}"
            raise TypeError(msg.format(type(target)))

        trans_geometry = transform_geometry_coord_system(self.data,
                                                         self.coord_system,
                                                         target_cs)
        return self.copy(new_geom=trans_geometry, new_cs=target_cs)

    def union(self, a_shape):
        """Perform the :py:mod:`shapely` union with another :class:`.Shape`
        object.

        Arguments:
            a_shape (:class:`Shape`): A shape

        Returns:
            The union of both :class:`.Shape` objects
        """
        return manipulate(self, 'union', a_shape)

    def x_rotate_cube(self, cube):
        """Rotate a cube's data **in place** along the longitude axis to the
        shape's bounds.

        Arguments:
            cube (:class:`iris.cube.Cube`): The cube to rotate
        """
        rotate_cube_longitude(self.data.bounds[0], cube)


# ------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------


def check_2d_latlon_cube(cube):
    """Check cube is a bidimensinal latitude/longitude slice.

    Arguments:
        cube (:class:`iris.cube.Cube`): The target cube

    Returns:
        `True` if `cube` is bidimensional and has latitude and longitude
        coordinates

    Raises:
        ValueError: If `cube` is not bidimensional
    """
    if not len(cube.shape) == 2:
        msg = "Cube must have only 2 dimensions, got shape {!r}"
        raise ValueError(msg.format(cube.shape))
    cube_primary_xy_coord_names(cube)
    return True


def check_cube_instance(cube):
    """Check an iris.Cube instance has been provided.

    Arguments:
        cube (:class:`iris.cube.Cube`): The cube to check

    Returns:
        `True` if the passed argument is a cube, `False` otherwise

    Raises:
        TypeError: If cube passed is not an Iris cube
    """
    if not isinstance(cube, Cube):
        msg = "Iris.Cube instance required, got {}"
        raise TypeError(msg.format(type(cube)))
    return True


def check_geometry_validity(geometry):
    """Checks the validity of a geometry.

    Arguments:
        geometry: A :class:`shapely.geometry.base.BaseGeometry` geometry
                  instance to check
    """
    try:
        res = geometry.is_valid
    except Exception as e:
        raise TypeError('Geometry is not a valid Shapely object')
    if not res:
        raise shapely.geos.TopologicalError('Not a valid geometry')
    return True


def check_intersection_cubestype(cubes):
    """Check if intersection cubes are present in a list.

    Arguments:
        cubes (list or :class:`iris.cube.CubeList`): A list of weights cubes

    Returns:
        `True` if ascend generated intersection cubes are found, `False`
        otherwise
    """
    for cube in cubes:
        if cube and cube.long_name == LONGNAME_INTERSECT:
            return True
    return False


def cluster_geometries(geometry, threshold=None, buffer_val=1.0):
    """Separates geometries into smaller clusters of geometries, where the
    geometries in each cluster are closer to each other than to those in
    other clusters.

    Arguments:
        geometry: A :class:`shapely.geometry.base.BaseGeometry` geometry
        threshold: the distance threshold separating clusters. If None, a
                   default threshold will be calculated based on the area
                   of the geometry
        buffer_val (float): the distance in shape units (normally degrees) to be
                            buffered
    """
    # let each geometry be represented by its central point
    centres = [np.array([geom.centroid.x, geom.centroid.y])
               for geom in geometry]
    if len(centres) == 1:
        # catch single geometries hidden in Multi geometries
        return [geometry]
    if not threshold:
        # set a default threshold - this is pretty arbitrary
        if isinstance(geometry, sgeom.MultiPolygon):
            threshold = geometry.area * 0.025
        else:
            threshold = geometry.buffer(buffer_val).area * 0.025
    clusters = hcluster.fclusterdata(np.array(centres),
                                     threshold,
                                     criterion="distance")
    # create a list to store the cluster geometries
    clustered_geoms = []
    cluster_indices = sorted(np.unique(clusters))
    if len(cluster_indices) == 1:
        # only one cluster
        return [geometry]
    else:
        for idx in cluster_indices:
            if not isinstance(geometry, sgeom.MultiLineString):
                # doesn't work for linestrings for some reason
                sub_geoms = np.array(geometry)[clusters == idx]
            else:
                sub_geoms = [geometry[count]
                             for count, ii in enumerate(clusters)
                             if ii == idx]
            clustered_geoms.append(type(geometry)([geom for geom
                                                   in sub_geoms]))
        return clustered_geoms

 
def compare_iris_cube_grids(cube1, cube2):
    """Compare the grids of two cubes.

    Arguments:
        cube1 (:class:`iris.cube.Cube`): An Iris Cube
        cube2 (:class:`iris.cube.Cube`): An Iris Cube

    Returns:
        `True` if the two cubes are on the same horizontal grid (the primary
        latitude and longitude points match as well as the coordinate systems),
        `False` otherwise
    """
    check_cube_instance(cube1)
    check_cube_instance(cube2)

    c1_lat_name, c1_lon_name = cube_primary_xy_coord_names(cube1)
    c1_lat_coord = cube1.coord(c1_lat_name)
    c1_lon_coord = cube1.coord(c1_lon_name)

    c2_lat_name, c2_lon_name = cube_primary_xy_coord_names(cube2)
    c2_lat_coord = cube2.coord(c2_lat_name)
    c2_lon_coord = cube2.coord(c2_lon_name)

    if (c1_lat_coord.points.shape == c2_lat_coord.points.shape and
            c1_lon_coord.points.shape == c2_lon_coord.points.shape):
        lat_match = np.allclose(c1_lat_coord.points, c2_lat_coord.points)
        lon_match = np.allclose(c1_lon_coord.points, c2_lon_coord.points)
        cs_match = get_cube_coord_system(cube1) == get_cube_coord_system(cube2)
        return all([lat_match, lon_match, cs_match])

    return False


def constrain_cube_expand(cube, argdict, latitude, longitude):
    """Constrain a cube after extending the required boundaries
    by one grid cell length in each direction.

    Arguments:
        cube           : The :class:`iris.cube.Cube` cube to constrain
        argdict (dict) : A dictionary of the form
                         {lat_name: (south, north), {lon_name}: (west, east)}
        latitude  (str): The latitude coordinate name or None
        longitude (str): The longitude coordinate name or None

    Returns:
        A cube constrained to the given bounds, or None
    """
    if latitude:
        argdict[latitude] = expand_range_with_coord(argdict[latitude],
                                                    cube.coord(latitude))
    if longitude:
        argdict[longitude] = expand_range_with_coord(argdict[longitude],
                                                     cube.coord(longitude))
    try:
        constrained_cube = cube.intersection(**argdict)
        return constrained_cube
    except Exception as e:  # TODO: precise exception (used to be "except:")
        return None


def constrain_cube_extract(cube, bounds, latitude, longitude):
    """Constrain a cube using basic iris constraints and cube.extract.

    Arguments:
        cube           : The :class:`iris.cube.Cube` cube to constrain
        bounds         : A 4-element :class:`numpy.array` array of (west,south,
                         east,north)
        latitude  (str): The latitude coordinate name or None
        longitude (str): The longitude coordinate name or None

    Returns:
        A cube constrained to the given bounds, or None
    """
    def lat_c(cell):
        return ((bounds[1] <= cell.point <= bounds[3]) or
                (cell.contains_point(bounds[1])) or
                (cell.contains_point(bounds[3])))

    def lon_c(cell):
        return ((bounds[0] <= cell.point <= bounds[2]) or
                (cell.contains_point(bounds[0])) or
                (cell.contains_point(bounds[2])))

    latc = iris.Constraint(**{latitude: lat_c})
    lonc = iris.Constraint(**{longitude: lon_c})
    try:
        icube = cube.extract(latc & lonc)
    except Exception as e:  # TODO: precise exception (used to be "except:")
        icube = None
    return icube


def constrain_cube_to_bounds(cube, bounds, latitude, longitude):
    """Constrain the given cube to a set of bounds.

    Arguments:
        cube           : The :class:`iris.cube.Cube` cube to constrain
        bounds         : A 4-element :class:`numpy.array` array of (west,south,
                         east,north)
        latitude  (str): The latitude coordinate name or None
        longitude (str): The longitude coordinate name or None

    Returns:
        A cube constrained to the given bounds
    """
    check_cube_instance(cube)
    coord_index = np.array([True, False, True, False])

    args = {}
    if latitude:
        args[latitude] = sorted(bounds[~coord_index])
    if longitude:
        args[longitude] = sorted(bounds[coord_index])

    # try intersecting as normal and catch some errors
    try:
        icube = cube.intersection(**args)

    # if it fails due to an indexing error then try expanding the
    # bounds by one lat/lon step in each direction and then performing
    # a normal constraint
    except IndexError as err:
        if VERBOSE:
            msg = ("Cube.intersection failed with IndexError; {} "
                   "--> using basic cube constraints")
            print(msg.format(err))
        icube = constrain_cube_expand(cube, args, latitude, longitude)
        if icube:
            icube = constrain_cube_extract(icube, bounds, latitude, longitude)

    # catch a value error
    except ValueError as err:
        if VERBOSE:
            print('Cube.intersection failed with ValueError; {} '
                  .format(err))

        # if it fails due to a value error then try using a simple
        # constraint method instead of cube.intersection
        if err.args[0] == ('coordinate units with no '
                           'modulus are not yet supported'):
            if VERBOSE:
                print("--> using basic cube constraints")
            icube = constrain_cube_extract(cube, bounds, latitude, longitude)

        # try removing the bounds and expanding the range
        else:
            if VERBOSE:
                print(('--> expanding boundaries and re-attempting '
                       'intersection, cube bounds are removed and replaced '
                       'with coord.guess_bounds()'))
            try:
                cube.coord(longitude).bounds = None
                cube.coord(latitude).bounds = None
                icube = constrain_cube_expand(cube, args, latitude, longitude)
                cube_xy_guessbounds(icube)
            except Exception as e:  # TODO: precise exception (used to be "except:")
                icube = None

    # if a different error is encountered then return None
    except Exception as e:  # TODO: precise exception (used to be "except:")
        if VERBOSE:
            print(('Cube.intersection failed with an error not yet handled, '
                   'returning None'))
        icube = None

    return icube


def create(coordinates, attributes, geom_type, *args, **kwargs):
    """Create a custom :class:`Shape` object using :py:mod:`shapely.geometry`.

    Arguments:
        coordinates      : The geometry coordinates as a list or tuple of xy
                           pairs (see :py:mod:`shapely.geometry` help)
        attributes (dict): A dictionary of metadata for the new :class:`.Shape`
                           object
        geom_type  (str) : A string of the geometry type, one of *Point*,
                           *LineString*, *LinearRing*, *Polygon*, *MultiPoint*,
                           *MultiLineString*, *MultiPolygon*
        args             : :py:mod:`shapely` arguments for the
                           :py:mod:`shapely.geometry` function associated with
                           the `geometry` to create
        kwargs           : :py:mod:`shapely` keyword arguments for the
                           :py:mod:`shapely.geometry` function associated with
                           the `geometry` to create

    Returns:
      A new :class:`Shape` object

    Examples:
        Create point (0.0, 0.0) for the city of Exeter

        >>> exeter = create((0.0, 0.0), {'city': 'Exeter'}, 'Point')

        Create a line for the Exe river

        >>> exe = create([(0.5, 3.0), (0.0, 0.0), (0.5, -2.0)],
        ...              {'river': 'Exe'}, 'LineString')

        Create a polygon for the Met Office building

        >>> met_office = create([(0, 0), (0, 1), (1, 1), (1, 0)],
        ...                     {'building': 'Met Office'}, 'Polygon')
    """

    geometries = ['Point', 'LineString', 'LinearRing', 'Polygon', 'MultiPoint',
                  'MultiLineString', 'MultiPolygon']

    # Argument types

    if geom_type not in geometries:
        raise ValueError('{} must be one of {}'
                         .format(geom_type, ', '.join(geometries)))

    if not isinstance(attributes, dict):
        raise TypeError('`attributes` must be a dictionary')

    islist = isinstance(coordinates, (list, tuple))
    coordinate_list = copy.deepcopy(coordinates) if islist else [coordinates]

    # Create shapely object

    func = getattr(sgeom, geom_type)
    obj = func(coordinate_list, *args, **kwargs)
    check_geometry_validity(obj)

    # Create class instance

    a_shape = Shape(obj, attributes.copy())

    # Return new shape

    return a_shape


def cube_bbox_shape_intersection(cube, a_shape):
    """Determines the intersection of a shape with the bounding
    box of a cube.

    Arguments:
        cube: An :class:`iris,.cube.Cube`, for which the bounds of
              the horizontal coordinates are to be determined
        a_shape: a Shape object, whose intersection with the cube bounding
               box is to be determined
    """
    # constrain cube to shape
    con_cube = a_shape.constrain_cube(cube)
    cube_cs = cube.coord_system()
    if not con_cube:
        return Shape(sgeom.Polygon(),
                     {'shape': 'empty'},
                     coord_system=cube_cs)

    # get cube bounds
    cube_bounds = get_cube_bounds(con_cube)

    # create a geometry from the bounds
    cube_bbox = sgeom.box(*cube_bounds)

    # project bounding box to that of shape
    cube_cs = get_cube_coord_system(con_cube)
    scs = a_shape.coord_system
    if not isinstance(scs, type(cube_cs)):
        shape_crs = scs.as_cartopy_projection()
        cube_crs = cube_cs.as_cartopy_crs()
        trans_bbox = shape_crs.project_geometry(cube_bbox, cube_crs)
    else:
        trans_bbox = cube_bbox

    # calculate the intersection of the shape and the
    # transformed bounding box
    cube_bbox_shape = Shape(trans_bbox, {'type': 'cube_shape_intersect'},
                            coord_system=cube_cs)
    bbox_intersect = a_shape.intersection(cube_bbox_shape)
    return bbox_intersect


def cube_primary_xy_coord_names(cube):
    """Return the primary latitude and longitude coordinate standard names, or
    long names, from a cube.

    Arguments:
        cube (:class:`iris.cube.Cube`): An Iris cube

    Returns:
        The names of the primary latitude and longitude coordinates
    """
    check_cube_instance(cube)
    latc = cube.coords(axis='y')[0] if cube.coords(axis='y') else -1
    lonc = cube.coords(axis='x')[0] if cube.coords(axis='x') else -1

    if -1 in (latc, lonc):
        msg = "Error retrieving xy dimensions in cube: {!r}"
        raise ValueError(msg.format(cube))

    latitude = latc.standard_name if latc.standard_name else latc.long_name
    longitude = lonc.standard_name if lonc.standard_name else lonc.long_name
    return latitude, longitude


def cube_xy_guessbounds(cube):
    """Guess latitude/longitude bounds of the cube and add them (**in place**)
    if not present.

    Arguments:
        cube (:class:`iris.cube.Cube`): An Iris cube

    Warning:
        This function modifies the passed `cube` in place, adding bounds to the
        latitude and longitude coordinates.
    """
    check_cube_instance(cube)
    for coord in cube_primary_xy_coord_names(cube):
        if not cube.coord(coord).has_bounds():
            cube.coord(coord).guess_bounds()


def determine_action(action, geom):
    """Work out whether `action` is a :py:mod:`shapely` object attribute, a
    :py:mod:`shapely` object method or a :py:mod:`shapely` function.

    Arguments:
        action (str)               : A :py:mod:`shapely` action
        geom   (`shapely.geometry`): A :py:mod:`shapely` geometry the `action`
                                     would be apply to (required to determine
                                     the nature of `action`)

    Returns:
        A string denoting the nature of the action, one of:

        * *attr* if the action is a :py:mod:`shapely` object attribute
        * *meth* if the action is a :py:mod:`shapely` object method
        * *func* if the action is a :py:mod:`shapely` object function

    Raises:
        ValueError: The nature of `action` could not be determined
    """

    if not isinstance(action, str):
        raise TypeError('`action` must be a string')
    try:
        check_geometry_validity(geom)
    except Exception as e:  # TODO: precise exception (used to be "except:")
        raise TypeError('`geom` is not a valid shapely geometry')

    # Notes on determining the nature of `action` in Python:
    # - inspect.isfunction(): works well but not if the function's module is
    #   not loaded already. Example: shapely.prepared.prep() is recognised as
    #   a function only if shapely.prepared was imported.
    # - inspect.ismethod(): does not work.
    # - method versus attribute: a dir(Shape) lists methods as attributes.

    action = action.strip()
    if '.' in action:
        nature = 'func'
    elif action in dir(geom) and hasattr(getattr(geom, action), '__call__'):
        nature = 'meth'
    elif action in dir(geom):
        nature = 'attr'
    else:
        raise ValueError('cannot work out what action `{}` is'.format(action))

    return nature


def expand_range_with_coord(a_range, coord):
    """Expand a value range by one grid cell length given an Iris cube
    coordinate.

    Arguments:
        a_range (tuple, list) : 2-element values to be expanded
        coord                 : An Iris Cube.coord

    Returns:
        An expanded range tuple (min-step, max+step)
    """
    dstep = np.max(np.abs(coord.points[1:] - coord.points[:-1]))
    return (np.min(a_range) - dstep, np.max(a_range) + dstep)


def geometry_factor(geom, sub_geom):
    """Calculates the proportion of a geometry a subgeometry comprises

    Arguments:
        geom: the whole geometry
        sub_geom: the subset of the whole geometry
    """
    sf = None
    if isinstance(sub_geom, sgeom.Point):
        sf = lambda g, sg: 1.0 / len(g) if hasattr(g, '__len__') else 1.0

    if isinstance(sub_geom, sgeom.MultiPoint):
        sf = lambda g, sg: len(sg) / float(len(g))

    if isinstance(sub_geom, (sgeom.LineString, sgeom.MultiLineString)):
        def sf(g, sg):
            if np.abs(g.length) > 1e-20:
                return sg.length / g.length
            return 1.0

    if isinstance(sub_geom, (sgeom.Polygon, sgeom.MultiPolygon)):
        sf = lambda g, sg: sg.area / g.area if np.abs(g.area) > 1e-20 else 1.0

    return sf(geom, sub_geom)


def get_cube_bounds(cube, expansion=1.0):
    """Determines the bounds of the horizontal coordinates of
    the given cube.

    Arguments:
        cube: An :class:`iris,.cube.Cube`, for which the bounds of
              the horizontal coordinates are to be determined
        expansion: a value by which to expand the bounds of a cube,
                   so that both end points of a grid are captured.
    """
    # get the primary coordinate names
    yname, xname = cube_primary_xy_coord_names(cube)
    # ensure coordinates have bounds
    for coord in (xname, yname):
        if not cube.coord(coord).has_bounds():
            cube.coord(coord).guess_bounds()
    # get the extents of the coordinates
    x_min = cube.coord(xname).bounds.min()
    x_max = cube.coord(xname).bounds.max()
    y_min = cube.coord(yname).bounds.min()
    y_max = cube.coord(yname).bounds.max()
    return np.array([x_min - expansion,
                     y_min - expansion,
                     x_max + expansion,
                     y_max + expansion])


def get_cube_coord_system(cube, default_cs=DEFAULT_CS):
    """Get a cube's coordinate system.

    Arguments:
        cube      : An :class:`iris.cube.Cube` cube
        default_cs: :class:`iris.coord_systems.CoordSystem` coordinate system
                    to be used if missing from cube

    Returns:
        The coordinate system (:class:`iris.coord_systems.CoordSystem`) of
        `cube`
    """
    check_cube_instance(cube)
    cube_cs = cube.coord_system()
    if not cube_cs:
        if VERBOSE:
            warnings.warn("Cube has no coord_system; using default lat/lon")
        cube_cs = default_cs
    return cube_cs


def get_shape_attributes(*args, **kwargs):
    """Returns a list of the `Shape.attributes` dictionaries of all
    :class:`.Shape` and :class:`.ShapeList` objects found in `args` and
    `kwargs` (if any).

    Arguments:
        args  : The arguments in which to search for :class:`.Shape` and
                :class:`.ShapeList` objects
        kwargs: The keyword arguments in which to search :class:`.Shape` and
                :class:`.ShapeList` objects

    Each `args` (tuple) and `kwargs` (dictionary) value can be :class:`.Shape`
    object, a :class:`.ShapeList` object, or a list/tuple/dict of
    :class:`.Shape` objects. Other occurrences of :class:`.Shape` or
    :class:`.ShapeList` objects will be ignored (examples: a list of
    :class:`.ShapeList`, a :class:`.Shape` nested more deeply into a
    list/tuple/dict, etc.).
    """

    if not isinstance(args, tuple):
        raise TypeError('`args` is not a tuple')
    if not isinstance(kwargs, dict):
        raise TypeError('`kwargs` is not a dictionary')

    dict_list = list()
    for i, arg in enumerate(args):
        if isinstance(arg, Shape):
            dict_list.append(copy.deepcopy(arg.attributes))
        elif isinstance(arg, (list, tuple, ShapeList)):
            for j, jarg in enumerate(arg):
                if isinstance(jarg, Shape):
                    dict_list.append(copy.deepcopy(jarg.attributes))
        elif isinstance(arg, dict):
            for key, value in arg.items():
                if isinstance(value, Shape):
                    dict_list.append(copy.deepcopy(value.attributes))

    for key, value in kwargs.items():
        if isinstance(value, Shape):
            dict_list.append(copy.deepcopy(value.attributes))
        elif isinstance(value, (list, tuple, ShapeList)):
            for j, jarg in enumerate(value):
                dict_list.append(copy.deepcopy(jarg.attributes))
        elif isinstance(value, dict):
            for kkey, vvalue in value.items():
                if isinstance(vvalue, Shape):
                    dict_list.append(copy.deepcopy(vvalue.attributes))

    return dict_list


def inspect_shp_attributes(shp_file, attributes):
    """Find the values of the given attributes in a shapefile.

    Arguments:
        shp_file   (str)        : The shapefile search
        attributes (str or list): An attribute or list of attributes for which
                                  to list the values

    Returns:
        A dictionary of the shapefile attributes and values
    """
    if not isinstance(attributes, str) and not isinstance(attributes, list):
        raise TypeError('`attributes` must be a string or list of strings')
    if isinstance(attributes, str):
        attributes = [attributes]
    result = dict()
    for attribute in attributes:
        records = shpreader.Reader(shp_file).records()
        func = lambda record: record.attributes[attribute]
        values = map(func, records)
        result[attribute] = sorted(list(set(values)))
    return result


def inspect_shp_info(shp_file, shp_version='.VERSION.txt'):
    """Describe a shapefile in terms of version and attributes.

    Arguments:
        shp_file    (str): The shapefile
        shp_version (str): The suffix of a version file associated with the
                           shapefile

    Returns:
        The file version and a list of attributes
    """
    version_file = '{}{}'.format(os.path.splitext(shp_file)[0], shp_version)
    version = 'unknown'
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            version = f.read()
    attrs = list()
    records = shpreader.Reader(shp_file).records()
    for record in records:
        attrs.extend(record.attributes.keys())
    return version, sorted(list(set(attrs)))


def inspect_shp_value(shp_file, value):
    """Find the corresponding attribute(s) for the given attribute value.

    Arguments:
        shp_file (str): The shapefile
        value         : A value to search for (can be string, float, etc.)

    Returns:
          List of attributes that hold the specified value (empty list if none
          found)
    """
    result = list()
    records = shpreader.Reader(shp_file).records()
    for record in records:
        for attribute, item_value in record.attributes.items():
            if value == item_value:
                result.append(attribute)
    return list(set(result))


def is_valid_Shape(source):
    """Check source is a valid :class:`.Shape` object.

    Arguments:
        source (object): The object that is to check

    Returns:
        `True` if the passed object is a :class:`.Shape` instance and is valid,
        `False` otherwise.
    """
    if isinstance(source, Shape) and source.is_valid:
        return True
    return False


def load_shp(files, coord_system=DEFAULT_CS, keep_invalid=False, **kwargs):
    """Read in a shapefile as a :class:`.ShapeList` object.

    Arguments:
        files              : A filename (str) or list of filenames to be loaded
        coord_system       : The :class:`iris.coord_systems.CoordSystem`
                             coordinate system of the object if known, else
                             defaults to regular latitude/longitude
        keep_invalid (bool): keep shape objects with invalid geometry?
        kwargs             : Filtering keyword arguments (see below)

    Returns:
        A list of shapes (:class:`.ShapeList`) contained in `files` and meeting
        the keyword filtering (if provided)

    See also:
        :meth:`Shape.filter` for usage of filtered loading.
    """
    shapes = ShapeList()
    if not isinstance(files, (list, tuple)):
        files = [files]
    for ifile in files:
        if not glob.glob(ifile):
            msg = "File does not exist: {!r}".format(ifile)
            warnings.warn(msg)
            continue
        sf = shpreader.Reader(ifile)
        for record in sf.records():
            a_shape = Shape(record.geometry, record.attributes,
                            coord_system=coord_system)
            if a_shape.is_valid or keep_invalid:
                shapes.append(a_shape)
            else:
                msg = "load_shp: Invalid geometry ignored; {}"
                print(msg.format(a_shape.summary(short=True)))
    return shapes.filter(**kwargs)


def manipulate(shapes, action, *args, **kwargs):
    """Manipulate a :class:`.Shape` or :class:`.ShapeList` object
    with :py:mod:`shapely` functionalities.

    Arguments:
        shapes      : The primary shape(s) to work with (a :class:`.Shape`, a
                      :class:`.ShapeList`, a list/tuple of :class:`.Shape`)
        action (str): The :py:mod:`shapely` function, method or attribute to
                      apply to `shapes`
        args        : :py:mod:`shapely` arguments for the action
        kwargs      : :py:mod:`shapely` keyword arguments for the action

    Returns:
        A new :class:`.Shape` object that is the result of `action` and has the
        additional attribute :py:attr:`manipulated` storing the history of
        manipulations applied to `shapes`.

    `args` and `kwargs` may contain :class:`.Shape` or :class:`.ShapeList`
    objects instead of :py:obj:`shapely.geometry` geometries, in which case
    their corresponding geometry will be automatically extracted for use with
    the action.

    **Action type**

    3 types of `action` are available: stand-alone :py:mod:`shapely` functions,
    :py:mod:`shapely` object methods or attributes (see below). Further
    action arguments (e.g. other geometries, parameters) must be specified via
    `args` and `kwargs` (these are passed straight on to the action). Note that
    `action` must result in a valid :obj:`shapely.geometry` geometry.

    **New object's attributes**

    If `shapes` is a :class:`.Shape` object, the new :class:`.Shape` object will
    inherit the :attr:`.attributes` of the :class:`.Shape` objects involved.
    If `shapes` is a :class:`.ShapeList` object or a list/tuple of
    :class:`.Shape` objects, the new :class:`.Shape` object will inherit
    :attr:`.attributes` of all the shapes in `shapes` (other :class:`.Shape`
    objects, e.g. in `args` or `kwargs`, will be ignored). See also
    :meth:`.Shape.append_attributes`.

    Examples:
        a_shape, new_shape, etc. are all Shape objects here. shape_list is a
        ShapeList object.

        1. Attribute action: retrieve the interior of a polygon shape

            new_shape = manipulate(a_shape, 'interior')
            # shapely equivalent of the above command:
            a_shape.data.interior

        2. Method action: do the union of two shapes

            new_shape = manipulate(shape_1, 'union', shape_2)
            # shapely equivalent of the above command:
            shape_1.data.union(shape_2.data)

        3. Method action: scale a geometry by a factor 2 in the x dimension

            new_shape = manipulate(a_shape, 'shapely.affinity.scale', xfact=2.0)
            # shapely equivalent of the above command:
            shapely.affinity.scale(a_shape.data, xfact=2.0)

        4. Function action: do the union of shapes

            new_shape = manipulate(shape_list, 'shapely.ops.cascaded_union')
            # shapely equivalent of the above command:
            shapely.ops.cascaded_union([x.data for x in shape_list])

            new_shape = manipulate([shape_1, shape_2, shape_3],
                                   'shapely.ops.cascaded_union')
            # shapely equivalent of the above command:
            shapely.ops.cascaded_union([shape_1.data, shape_2.data,
                                       shape_3.data])
    """

    # CHECK FUNCTION ARGUMENTS

    # Argument type

    if not isinstance(shapes, (Shape, ShapeList, list, tuple)):
        raise TypeError('`shapes` must be a Shape object, a ShapeList object or'
                        ' a list/tuple of Shape objects')
    if isinstance(shapes, (ShapeList, list, tuple)):
        for item in shapes:
            if not isinstance(item, Shape):
                raise TypeError('`shapes` must be a Shape object, a ShapeList'
                                ' object or a list/tuple of Shape objects')

    if isinstance(shapes, (ShapeList, list, tuple)):
        result = [is_valid_Shape(item) for item in shapes]
    else:
        result = [is_valid_Shape(shapes)]
    if False in result:
        raise TypeError('`shapes` is or contains an invalid Shape')

    # Prepare shapely action's arguments
    # in shapes/args/kwargs by their corresponding shapely geometries (i.e.
    # Shape.data) since shapely does not know what a Shape/ShapeList is.

    if isinstance(shapes, (ShapeList, list, tuple)):
        action_geom = [item.data for item in shapes]
        coord_systems = [item.coord_system for item in shapes]
        if any([x != coord_systems[0] for x in coord_systems]):
            msg = ("Warning, found {} number of coord systems;"
                   " please check output carefully")
            print(msg.format(sum(1 + len([x != coord_systems[0]
                                          for x in coord_systems]))))
    else:
        action_geom = shapes.data
    action_args, action_kwargs = shapelify_args(*args, **kwargs)

    # Determine nature of the action: a shapely function, geometry method or
    # geometry attribute? (need a shapely geometry to answer this)

    action = action.strip()
    if isinstance(shapes, Shape):
        action_type = determine_action(action, shapes.data)
    elif isinstance(shapes, (ShapeList, list, tuple)):
        action_type = determine_action(action, shapes[0].data)
    else:
        action_type = None

    # Check compatibility of given shape with action (note: shapes is restricted
    # to a Shape or ShapeList although any of Shape/ShapeList/shapely
    # geometries can be passed to the action via args and kwargs)

    if (action_type in ['meth', 'attr'] and
            isinstance(shapes, (ShapeList, list, tuple))):
        raise TypeError('action `{}` cannot work on a ShapeList object or '
                        'list/tuple of Shape objects'.format(action))
    if action_type == 'attr' and (len(args) != 0 or len(kwargs) != 0):
        raise ValueError('action `{}` does not take in any argument or keyword'
                         ' argument'.format(action))

    # APPLY ACTION TO SHAPE

    geom = None
    try:
        if action_type == 'func':
            module_name, function_name = action.rsplit('.', 1)
            module = importlib.import_module(module_name)
            func = getattr(module, function_name)
            geom = func(action_geom, *action_args, **action_kwargs)
        elif action_type == 'meth':
            method = getattr(action_geom, action)
            geom = method(*action_args, **action_kwargs)
        elif action_type == 'attr':
            geom = getattr(action_geom, action)
    except Exception as e:
        raise e

    # Shape attributes handling. If `shapes` is:
    # - Shape => new_shape.attributes = .attributes of all Shape objects
    #   provided (shapes + any in args + any in kwargs)
    # - ShapeList or list/tuple of Shape objects => new_shape.attributes =
    #   .attributes of all shapes in ShapeList/list/tuple (those in args/kwargs
    #   are ignored if any)
    if isinstance(shapes, Shape):
        new_attr = get_shape_attributes(shapes, *args, **kwargs)
        cs = shapes.coord_system
    elif isinstance(shapes, (ShapeList, list, tuple)):
        new_attr = get_shape_attributes(shapes)
        cs = shapes[0].coord_system
    else:
        new_attr = None
        cs = None

    # TRANSFER RESULT TO A NEW SHAPE OBJECT
    new_shape = Shape(geom, dict(), coord_system=cs)
    new_shape.append_attributes(new_attr)

    # RECORD ACTION APPLIED

    if 'manipulated' in new_shape.attributes.keys():
        value = new_shape.attributes['manipulated']  # past shape manipulations
        new_shape.attributes['manipulated'] = '{}, {}'.format(value, action)
    else:
        new_shape.attributes['manipulated'] = str(action)

    # RETURN NEW SHAPE

    if not is_valid_Shape(new_shape):
        warnings.warn('The result of manipulate is not a valid Shape')

    return new_shape


def mask_cube_with_minimum_weights(cube, weights, minimum_weight):
    """Returns the mask of the intersection between the given cube and polygon.

    Arguments:
        cube                  : An :class:`iris.cube.Cube` cube
        weights               : A bidimensional (latitude/longitude)
                                :class:`numpy.array` array of weights to be
                                broadcast across `cube.data`
        minimum_weight (float): value to be masked

    Warning:
        The masking is performed in place.
    """
    check_cube_instance(cube)
    mask = np.ma.masked_less(weights, minimum_weight)
    combined_mask = np.ma.getmaskarray(cube.data) + np.ma.getmaskarray(mask)
    cube.data = np.ma.MaskedArray(cube.data, mask=combined_mask)
    cube.attributes['minimum_weighting'] = minimum_weight


def plot(shapes, ax=None, point_buffer=0.25, projection=ccrs.PlateCarree(),
         **kwargs):
    """Add shapes to cartopy axes (or create a new one).

    Arguments:
        shapes     : An iterable of :class:`.Shape` objects (a list/tuple of
                     shapes or a :class:`ShapeList` object)
        ax         : An Iris/Cartopy axes object for the shapes to be added to
        point_buffer : The value to buffer Point or MultiPoint shapes with
        projection : Cartopy projection of the map (default==PlateCarree)
    """
    shapelist = shapes if hasattr(shapes, '__iter__') else [shapes]
    kwargs_cp = {}
    for key, val in kwargs.items():
        if hasattr(val, '__iter__'):
            if len(val) == len(shapelist):
                kwargs_cp[key] = val
            elif len(val) == 1:
                kwargs_cp[key] = val[0]
            else:
                n_tile = int(np.ceil(float(len(shapelist)) / len(val)))
                kwargs_cp[key] = list(np.tile(val, n_tile)[:len(shapelist)])
        else:
            kwargs_cp[key] = val

    # if no axes provided then get/create one
    if ax is None:
        ax = plt.gca()
        iplt._replace_axes_with_cartopy_axes(projection)
        ax = plt.gca()

    # plot the shapes
    for idx, a_shape in enumerate(shapelist):
        crs = a_shape.coord_system.as_cartopy_projection()
        if isinstance(a_shape.data, (sgeom.Point, sgeom.MultiPoint)):
            geom = a_shape.data.buffer(point_buffer)
        else:
            geom = a_shape.data

        sub_kwargs = {}
        for key, val in kwargs_cp.items():
            sub_kwargs[key] = val[idx] if hasattr(val, '__iter__') else val

        ax.add_geometries([geom], crs, **sub_kwargs)

    return ax


def rebase_coordinate_points(coord):
    """Rebase the points of a cube coordinate to it's modulus.

    Arguments:
        coord: A :class:`iris.coord` instance

    Returns:
        A numpy array of the rebased coordinate points
    """
    points = coord.points
    cmod = coord.units.modulus
    if cmod:
        points = rebase_values_to_modulus(points, cmod)
    return np.array(points)


def rebase_values_to_modulus(values, modulus):
    """Rebase values to a modulus value.

    Arguments:
        values (list, tuple): The values to be re-based
        modulus             : The value to re-base to

    Returns:
        A list of re-based values.
    """
    rebased = []
    for val in values:
        nv = np.abs(val) % modulus
        if val < 0.:
            nv *= -1.
        if np.abs(nv - modulus) < 1e-10:
            nv = 0.
        rebased.append(nv)
    return rebased


def reintersect_2d_indices(master_cube, sub_cube):
    """Get indexes for re-inserting a cube.

    Arguments:
        master_cube (:class:`iris.cube.Cube`): The main Iris cube to have data
                                               inserted
        sub_cube    (:class:`iris.cube.Cube`): The cube to be placed in the
                                               master cube

    Returns:
        ylocs, xlocs; lists of the y and x dimension indexes

    Warning:
        This operation only works on bidimensional cubes.

    """
    my_name, mx_name = cube_primary_xy_coord_names(master_cube)
    sy_name, sx_name = cube_primary_xy_coord_names(sub_cube)

    master_xs = rebase_coordinate_points(master_cube.coord(mx_name))
    master_ys = rebase_coordinate_points(master_cube.coord(my_name))
    sub_xs = rebase_coordinate_points(sub_cube.coord(sx_name))
    sub_ys = rebase_coordinate_points(sub_cube.coord(sy_name))

    dx = np.abs(0.05 * (master_xs[1] - master_xs[0]))
    xlocs = [np.where(np.abs(master_xs - xl) < dx)[0][0] for xl in sub_xs]
    dy = np.abs(0.05 * (master_ys[1] - master_ys[0]))
    ylocs = [np.where(np.abs(master_ys - yl) < dy)[0][0] for yl in sub_ys]

    return ylocs, xlocs


def reintersect_2d_slice(master_cube, sub_cube, roll_lon=False):
    """Spatially insert one bidimensional cube into another bidimensional cube.

    Arguments:
        master_cube (:class:`iris.cube.Cube`): The main Iris cube to have data
                                               inserted
        sub_cube    (:class:`iris.cube.Cube`): The cube to be placed in the
                                               master cube
        roll_lon (bool): If `True`, the longitudes are rotated

    Returns:
        A copy of the `master_cube` with the `sub_cube` inserted

    Warning:
        This operation only works on bidimensional cubes.
    """
    check_cube_instance(master_cube)
    check_cube_instance(sub_cube)
    check_2d_latlon_cube(master_cube)
    check_2d_latlon_cube(sub_cube)

    # get x/y coordinate names and point values

    y_name, x_name = cube_primary_xy_coord_names(master_cube)
    y_name_sub, x_name_sub = cube_primary_xy_coord_names(sub_cube)
    xs = master_cube.coord(x_name).points
    mcube = master_cube.copy()

    # determine whether to do nothing, roll
    # the longitude coordinate or try
    # cube.intersection

    dx = 0.05 * (xs[1] - xs[0])
    args = {x_name: [xs[0] - dx, xs[-1] + dx]}
    xmod = master_cube.coord(x_name).units.modulus
    sx = sub_cube.copy()
    if xmod is not None:
        if roll_lon:
            rotate_cube_longitude(xs[0], sx)
        else:
            sub_cube.coord(x_name).bounds = None
            sx = sub_cube.intersection(**args)

    # get re-intersection indexes

    ylocs, xlocs = reintersect_2d_indices(mcube, sx)
    xindx = xlocs * len(ylocs)
    yindx = np.repeat(ylocs, len(xlocs))

    # transpose subcube if needed

    mydim = mcube.coord_dims(y_name)[0]
    sydim = sx.coord_dims(y_name_sub)[0]
    if mydim != sydim:
        sx.transpose([1, 0])

    # re insert data at required indexes.

    if mydim == 0:
        mcube.data[yindx, xindx] = sx.data.flatten()
    elif mydim == 1:
        mcube.data[xindx, yindx] = sx.data.flatten()
    return mcube


def remove_non_lat_lon_cube_coords(cube, xy_coordinates=None):
    """Remove **in place** the non-primary latitude/longitude coordinates from a
    cube.

    Arguments:
        cube (:class:`iris.cube.Cube`): The target cube
        xy_coordinates (list or tuple): A 2-element list/tuple of the primary
                                        coordinate names (if not set then the
                                        primary coordinate names are determined
                                        from `cube`)

    Warning:
        This function removes any secondary latitude and longitude coordinates
        from `cube.`
    """
    if (not xy_coordinates or
            not isinstance(xy_coordinates, tuple) or
            not len(xy_coordinates) == 2):
        xy_coordinates = cube_primary_xy_coord_names(cube)

    for dim in ['x', 'y']:
        for coord in cube.coords(axis=dim):
            if (coord.standard_name not in xy_coordinates and
                    coord.long_name not in xy_coordinates):
                cube.remove_coord(coord)


def rotate_cube_longitude(x_origin, cube):
    """Rotate a cube's data **in place** along the longitude axis to the new
    longitude origin.

    Arguments:
        x_origin (float): The new x-coord origin
        cube            : The :class:`iris.cube.Cube` cube to rotate
    """
    check_cube_instance(cube)
    latitude, longitude = cube_primary_xy_coord_names(cube)
    xcord = cube.coord(longitude)
    dx = xcord.points[1] - xcord.points[0]
    lon_shift = xcord.points[0] - x_origin
    number_moves = np.int(np.rint(lon_shift / dx))
    new_coord = xcord.points - (number_moves * dx)
    cube.data = np.roll(cube.data, -number_moves,
                        axis=cube.coord_dims(longitude)[0])
    xcord.points = new_coord
    if xcord.has_bounds():
        xcord.bounds = xcord.bounds - (number_moves * dx)
    else:
        xcord.guess_bounds()


def save_shp(source, target, filetype, overwrite=False):
    """Save :class:`.Shape` or :class:`.ShapeList` objects to file
    (automatically creates the *.shp*, *.shx* and *.dbf* files).

    Arguments:
        source          : :class:`.Shape` or :class:`ShapeList` object(s) to
                          save
        target    (str) : The output file path (extension not required)
        filetype  (str) : Select *point*, *polyline* or *polygon* types
        overwrite (bool): Whether to overwrite any existing file named `target`
                          (with extensions *.shp*, *.shx* and *.dbf*)

    Raises:
        IOError   : If `overwrite=False` and the target files already exist
                    (error raised and no file is overwritten)
        ValueError: If `filetype` is not one of *point*, *polyline*, *polygon*

    Note:
        All shapes must have the same attributes and geometry type; this is a
        limitation of the shapefile format.
    """
    for prefix in ['', '.shp', '.dbf', '.shx']:
        if glob.glob(target + prefix) and not overwrite:
            msg = '{!r} already exists, delete original before saving...'
            raise IOError(msg.format(target))
    if not isinstance(source, (list, tuple, ShapeList)):
        shapelist = ShapeList([source])
    else:
        shapelist = source
    keys = shapelist[0].attributes.keys()
    if filetype.lower() not in SHAPEFILE_TYPES.keys():
        msg = "{!r} is not a support filetype; valid types are {!r}"
        raise ValueError(msg.format(filetype, SHAPEFILE_TYPES.keys()))
    shapefile_type, wfunc = SHAPEFILE_TYPES[filetype.lower()]
    w = shapefile.Writer(shapefile_type)

    # add file field definitions

    for key in keys:
        val_type = type(shapelist[0].attributes[key])
        kwargs = SHAPEFILE_FIELD_TYPES[val_type]
        w.field(key, **kwargs)

    # add shapes and records

    for a_shape in shapelist:
        wfunc(w, parts=a_shape.get_coordinates_as_list())
        w.record(**a_shape.attributes)

    w.save(target)
    return True


def shapelify_args(*args, **kwargs):
    """Returns `args` and `kwargs` with occurrences of :class:`.Shape` or
    :class:`.ShapeList` replaced by their actual :py:mod:`shapely` geometries
    (i.e. their corresponding :attr:`.data` attribute).

    Arguments:
        args  : The arguments to check
        kwargs: The keyword arguments to check

    Each `args` (tuple) and `kwargs` (dictionary) value can be :class:`.Shape`
    object, a :class:`.ShapeList` object, or a list/tuple/dict of
    :class:`.Shape` objects. Other occurrences of :class:`.Shape` or
    :class:`.ShapeList` objects will be ignored (examples: a list of
    :class:`.ShapeList`, a :class:`.Shape` nested more deeply in a
    list/tuple/dict, etc.).
    """

    # When using *args or **kwargs in a function definition, args are returned
    # returned as a tuple and kwargs as a dictionary:
    #   >>> def f(a, *args, **kwargs):
    #   ...     print a, type(a)
    #   ...     print args, type(args)
    #   ...     print kwargs, type(kwargs)
    #   ...     return
    #   ...
    #   >>> f(1, 2, 3, b=4, **{'c': 'c', 'd': 'd'})
    #   1 <type 'int'>
    #   (2, 3) <type 'tuple'>
    #   {'c': 'c', 'b': 4, 'd': 'd'} <type 'dict'>
    #

    if not isinstance(args, tuple):
        raise TypeError('`args` is not a tuple')
    if not isinstance(kwargs, dict):
        raise TypeError('`kwargs` is not a dictionary')

    args2 = list(copy.deepcopy(args))
    for i, arg in enumerate(args):
        if isinstance(arg, Shape):
            if is_valid_Shape(arg):
                args2[i] = arg.data
            else:
                raise ValueError('invalid Shape found')
        elif isinstance(arg, (list, tuple, ShapeList)):
            if isinstance(arg, tuple):  # tuples are immutable
                args2[i] = list(args2[i])
            for j, jarg in enumerate(arg):
                if isinstance(jarg, Shape):
                    if is_valid_Shape(jarg):
                        args2[i][j] = jarg.data
                    else:
                        raise ValueError('invalid Shape found')
        elif isinstance(arg, dict):
            for key, value in arg.items():
                if isinstance(value, Shape):
                    if is_valid_Shape(value):
                        # Need deepcopy for this to work
                        args2[i][key] = value.data
                    else:
                        raise ValueError('invalid Shape found')

    kwargs2 = copy.deepcopy(kwargs)
    for key, value in kwargs.items():
        if isinstance(value, Shape):
            if is_valid_Shape(value):
                kwargs2[key] = value.data
            else:
                raise ValueError('invalid Shape found')
        elif isinstance(value, (list, tuple, ShapeList)):
            if isinstance(value, tuple):  # tuples are immutable
                kwargs2[key] = list(value)
            for j, jarg in enumerate(value):
                if is_valid_Shape(jarg):
                    kwargs2[key][j] = jarg.data
        elif isinstance(value, dict):
            for kkey, vvalue in value.items():
                if isinstance(vvalue, Shape):
                    if is_valid_Shape(vvalue):
                        # Need deepcopy for this to work
                        kwargs2[key][kkey] = vvalue.data
                    else:
                        raise ValueError('invalid Shape found')

    return args2, kwargs2


def show(shapes, bounds=False, projection=ccrs.PlateCarree(),
         scale='110m', facecolor='red'):
    """Show a simple visualisation of shapes.

    Arguments:
        shapes     : An iterable of :class:`.Shape` objects (a list/tuple of
                     shapes or a :class:`ShapeList` object)
        bounds     : `True` to plot extent taken from shape, `False` to use
                     global bounds, or a 4-element list of x0, x1, y0, y1
        projection : A :class:`cartopy.crs.CRS` map projection
        scale (str): Natural Earth map resolution ('10m', '50m', '110m')
        facecolor  : A string/list of color values (e.g. 'red', 'k', '#C0C0C0')
    """
    valid_scales = ('10m', '50m', '110m')
    if scale not in valid_scales:
        msg = "Invalid scale value {}, using '10m'".format(scale)
        warnings.warn(msg)
        scale = valid_scales[0]

    shapelist = shapes if hasattr(shapes, '__iter__') else [shapes]
    ax = plot(shapelist, facecolor=facecolor, projection=projection)
    if scale == '110m':
        ax.add_feature(cartopy.feature.BORDERS)
        ax.add_feature(cartopy.feature.COASTLINE)
    else:
        borders = NaturalEarthFeature(category='cultural',
                                      name='admin_0_countries',
                                      scale=scale, facecolor='none')
        ax.add_feature(borders)

    if bounds:
        if hasattr(bounds, '__len__'):
            ax.set_extent(bounds, crs=projection)
        else:
            bounds = np.array([a_shape.data.bounds for a_shape in shapelist])
            x_min = bounds.min(axis=0)[0]
            x_max = bounds.max(axis=0)[2]
            y_min = bounds.min(axis=0)[1]
            y_max = bounds.max(axis=0)[3]
            crs = shapelist[0].coord_system.as_cartopy_projection()
            ax.set_extent([x_min, x_max, y_min, y_max], crs=crs)
    else:
        ax.set_global()

    plt.show()
    plt.close()


def transform_geometry_coord_system(geometry, source, target):
    """Transform a geometry into a new coordinate system.

    Arguments:
        geometry: A :class:`shapely.geometry.base.BaseGeometry` *Polygon* or
                  *MultiPolygon* object
        source  : A geometry's :class:`iris.coord_systems.CoordSystem`
                  coordinate system
        target  : The geometry's new :class:`iris.coord_systems.CoordSystem`
                  coordinate system

    Returns:
        A transformed *Polygon*/*Multipolygon* instance

    Warning:
        This operation may result in geometries which are not valid. Please
        check that the transformed geometry is as you expect.
    """
    check_geometry_validity(geometry)
    if source == target:
        return geometry

    if hasattr(geometry, 'geoms'):
        trans_geoms = []
        for poly in geometry:
            trans_geoms.append(transform_geometry_points(poly, source, target))
        geo_type = type(geometry)
        trans_geometry = geo_type(trans_geoms)
    else:
        trans_geometry = transform_geometry_points(geometry, source, target)
    return trans_geometry


def transform_geometry_points(geometry, source, target):
    """Transform geometry points into a new coordinate system.

    Arguments:
        geometry: A :class:`shapely.geometry.base.BaseGeometry` geometry
        source  : A shape's :class:`iris.coord_systems.CoordSystem` coordinate
                  system
        target  : The new :class:`iris.coord_systems.CoordSystem` coordinate
                  system

    Returns:
        The points of the `geometry` in the `target` coordinate system.

    Note:
        Multi- and collection types are not supported (see
        :func:`transform_geometry_coord_system`).

    Warning:
        This operation may result in geometries which are not valid. Please
        check that the transformed geometry is as you expect.
    """
    target_proj = target.as_cartopy_projection()
    source_proj = source.as_cartopy_projection()
    fn = lambda xs, ys: target_proj.transform_points(source_proj,
                                                     np.array(xs, copy=False),
                                                     np.array(ys, copy=False)
                                                     )[:, 0:2].T
    return shapely.ops.transform(fn, geometry)


def update_progress(progress, msg, stdout):
    """Displays the progress of the current process

    Arguments:
        progress: a float in [0, 1] representing the progress made (progress %)
        msg: the message to be printed
        stdout: where the progress bar should be written
    """
    bar_length = 10  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 100:
        progress = 100
        status = "Done...\r\n"
    block = int(round(bar_length*progress/100.0))
    tf = ": [{0}] {1}% {2}"
    text = "\r" + msg + tf.format("#"*block + "-"*(bar_length-block),
                                  progress, status)
    stdout.write(text)
    stdout.flush()


def zero_cube(cube, inplace=False):
    """Replace a cube's data with zeroes.

    Arguments:
        cube          : An :class:`iris.cube.Cube` cube
        inplace (bool): Whether to perform the changes inplace

    Returns:
        A copy of the cube with zero data if `inplace` is `False`, nothing
        otherwise
    """
    zeros = np.zeros(cube.shape, dtype=np.float)
    if inplace:
        cube.data = zeros
    else:
        tcube = cube.copy()
        tcube.data = zeros
        return tcube

