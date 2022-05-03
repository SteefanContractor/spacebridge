import logging
import netCDF4 as nc
import numpy as np
import glob
import multiprocessing as mp
import os.path
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

"""
eg
----------------------

"""

BAD_VALUE = -999
GRZICE_BANDS = ['l1', 'l2']


def get_file_list_local(path_list):
    if isinstance(path_list, str):
        path_list = [path_list]

    files = []
    for path in path_list:
        files.extend(glob.glob(path))

    return sorted(list(set(files)))


def gpstime2utc_fast(time_gps_seconds):
    """
    Convert gps seconds to UTC datetime, see:
    https://racelogic.support/01VBOX_Automotive/01General_Information/Knowledge_Base/What_are_GPS_Leap_Seconds%3F#

    Parameters
    ----------
    time_gps_seconds : float
        GPS time

    Returns
    ----------
    utctime : datetime
        Datetime object in UTC.

    """
    utctime = datetime(1980, 1, 6, 0, 0, 0, 0, None) + timedelta(
        seconds=time_gps_seconds
    )
    if utctime > datetime(2016, 12, 31):
        leap_second = 18
    elif utctime > datetime(2015, 7, 1):
        leap_second = 17
    elif utctime > datetime(2012, 7, 1):
        leap_second = 16
    elif utctime > datetime(2009, 1, 1):
        leap_second = 15
    elif utctime > datetime(2006, 1, 1):
        leap_second = 14
    elif utctime > datetime(1999, 1, 1):
        leap_second = 13
    elif utctime > datetime(1997, 7, 1):
        leap_second = 12
    elif utctime > datetime(1996, 1, 1):
        leap_second = 11
    elif utctime > datetime(1994, 7, 1):
        leap_second = 10
    elif utctime > datetime(1993, 7, 1):
        leap_second = 9
    elif utctime > datetime(1992, 7, 1):
        leap_second = 8
    elif utctime > datetime(1991, 1, 1):
        leap_second = 7
    elif utctime > datetime(1990, 1, 1):
        leap_second = 6
    elif utctime > datetime(1988, 1, 1):
        leap_second = 5
    elif utctime > datetime(1985, 7, 1):
        leap_second = 4
    elif utctime > datetime(1983, 7, 1):
        leap_second = 3
    elif utctime > datetime(1982, 7, 1):
        leap_second = 2
    elif utctime > datetime(1981, 7, 1):
        leap_second = 1
    else:
        leap_second = 0
    utctime = utctime - timedelta(seconds=leap_second)
    return utctime


class LoadGNSSR:

    def __init__(self, path_to_dir, multi_processing=1, band='l2'):
        """
        :param path_to_dir string or list of strings pointing to the directories using globbing,
        eg. '[path/to/dir/*.nc', '/path/to/other/dir/*T0[1-5]*.nc']
        :param multi_processing: number of processes to run simultaneously
        """
        self.fig_fold = os.path.dirname(__file__)
        self.l1_path = path_to_dir
        self.l1_in_s3 = True
        self.override_flags = False
        self.grzice_band = band

        logging.basicConfig(level=logging.INFO,
                            format='[ %(asctime)s.%(msecs)03d ] %(levelname)s %(module)s - %(funcName)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger("Load_GNSSR")
        self.l1b_files = []
        self.data = {'files_read': []}
        self.multi_process_count = multi_processing

        self.vars = [
            'time',
            'longitude',
            'latitude',
            'reflectivity',
            'excess_phase_noise',
            'sea_ice_presence',
            'sea_ice_type',
            'phase_noise',
            'flag_land',
            'flag_high_elevation',
            'flag_antenna_gain',
            'flag_master',
            'file_ind'
        ]
        self.db_vars = [
            'reflectivity',
            'power_reflected',
            'power_direct',
            'snr_reflected',
            'snr_direct',
            'antenna_gain_reflected',
            'antenna_gain_direct'
        ]
        self.flag_vars = [
            'flag_master',
            'flag_low_elevation',
            'flag_high_elevation',
            'flag_land',
            'flag_antenna_gain',
            'flag_climatology',
            'flag_refl_in_prompt_tap'
        ]
        # Here are all the variables in the netcdf file. I have currently removed the 50 Hz
        # variables as we do not have flagging in for those.
        self.all_vars = [
                         'physical_antenna_name',
                         'rpa_virtual_antenna_id',
                         'dpa_virtual_antenna_id',
                         'flag_refl_in_prompt_tap',
                         'flag_navbits_not_removed',
                         'history',
                         'fill_value',
                         'leo_sat',
                         'occultation_sat',
                         'gps_seconds_start',
                         'gps_seconds_stop',
                         # 'time_50Hz',
                         'time',
                         'sea_ice_presence',
                         'sea_ice_type',
                         'reflectivity',
                         'snr_reflected',
                         'snr_direct',
                         # 'snr_l1b_reflected',
                         # 'snr_l1b_direct',
                         # 'residual_phase_l1b',
                         'phase_noise',
                         'excess_phase_noise',
                         'power_reflected',
                         'power_direct',
                         'antenna_gain_reflected',
                         'antenna_gain_direct',
                         'noise_rx',
                         'noise_l1b',
                         'rx_pos',
                         'tx_pos',
                         'spec_pos',
                         'angle_of_elevation',
                         'longitude',
                         'latitude',
                         'flag_land',
                         'flag_high_elevation',
                         'flag_low_elevation',
                         'flag_antenna_gain',
                         'flag_climatology',
                         'flag_master'
                         ]
        self.lat_str = 'latitude'
        self.lon_str = 'longitude'
        self.time_str = 'time'
        self.error_code = 0
        self._fillValue = BAD_VALUE
        self.points_added = 0

        # grid corners for gridgrid
        self.map_reduce_x = {'north': [-3850, 3750], 'south': [-3950, 3950]}
        self.map_reduce_y = {'north': [-5350, 5850], 'south': [-3950, 4350]}  # km

    def add_all_vars(self):
        self.vars = self.all_vars

    def add_var(self, new_var, convert_db=False):
        """
        Adds whichever var you want to the list of those to be collected up, with option to convert
        to dB.
        :param new_var: string, variable name to be added, identical to l1 netcdf name please
        :param convert_db: bool, whether it will need to be converted to dB
        :return: None
        """
        if new_var in self.vars:
            self.logger.info(f'{new_var} already in variable list, not re-adding.')
        else:
            self.vars.append(new_var)

        if convert_db:
            if new_var in self.db_vars:
                self.logger.info(f'{new_var} already in decibel conversion list, not re-adding.')
            else:
                self.db_vars.append(new_var)

    def collect_tracks(self, override_flags=False):
        """
        Collects up all the data from the files found in the l1_path, using the variables in the vars
        list into a nice, easy to use dictionary for whatever you like.
        Puts these in dictionary "self.data"
        Also, careful how many files you try to read in at once, yadda yadda yadda

        :param: override_flags - boolean, whether you wish to disregard the default flags and instead
        load all of the points in for your own filtering.

        :return: None
        """
        if override_flags:
            self.override_flags = True

        if isinstance(self.l1_path, str):
            self.l1_path = [self.l1_path]

        if override_flags:
            for flag in self.flag_vars:
                self.add_var(flag)

        self.l1b_files = get_file_list_local(self.l1_path)

        self.logger.info(f'Loading data from {len(self.l1b_files)} files')
        if self.multi_process_count > 1:
            pool = mp.Pool(self.multi_process_count)
            dicts_out = pool.map_async(self.read_data, range(0, len(self.l1b_files)))
            dicts_out = dicts_out.get()
            pool.close()
        else:
            dicts_out = []
            for i_file in range(0, len(self.l1b_files)):
                dicts_out.append(self.read_data(i_file))

        start_files_in = len(self.data['files_read'])
        for file in dicts_out:
            first_file_loaded_in = (self.points_added == 0)
            if not file:
                continue
            self.points_added += len(file[self.vars[0]])
            for key in file.keys():
                if key not in self.data:
                    self.data[key] = []
                if 'files_ind' in key:
                    self.data[key] = self.data[key] + start_files_in
                if first_file_loaded_in:  # if this is the first lot of tracks we have collected up
                    if 'files_read' in key:
                        self.data[key] = [file[key]]
                    else:
                        self.data[key] = np.array(file[key])
                else:
                    if 'files_read' in key:
                        self.data[key].append(file[key])
                    else:
                        self.data[key] = np.concatenate((np.array(self.data[key]), np.array(file[key])))

        self.logger.info('Data Loaded')
        if self.points_added == 0:
            self.logger.info('No data found that meets quality flags')
            self.error_code = 101
            return

    def read_data(self, i_file):
        temp_dict = {}
        local_file = self.l1b_files[i_file]

        # progress counter doesn't work so well with the multiprocessing
        if (self.multi_process_count == 1) & (np.remainder(i_file, 10) == 0):
            self.logger.info(
                f"Reading in file {i_file} of {len(self.l1b_files)} - {i_file / len(self.l1b_files) * 100:0.05} %")

        opened = nc.Dataset(local_file)

        if self.lat_str not in opened.variables.keys():
            return

        i_band = GRZICE_BANDS.index(self.grzice_band)

        good = self.get_good_indices(opened)

        if (np.count_nonzero(good) < 1) and ~self.override_flags:
            return

        with np.warnings.catch_warnings():
            np.warnings.simplefilter("ignore", category=RuntimeWarning)
            for var in self.vars:
                if var not in temp_dict:
                    temp_dict[var] = []

                if 'file_ind' in var:
                    value = i_file
                elif 'leo_sat' in var:
                    value = int(opened.__getattr__(var))
                elif 'rx_id' in var:
                    value = int(opened.__getattr__(var))
                elif 'flag_refl_in_prompt_tap' in var:
                    value = int(opened.__getattr__(var)[i_band])
                elif ('noise_rx' in var) or ('noise_l1b' in var):
                    value = opened[var][i_band]
                elif var not in opened.variables.keys():
                    if var in opened.ncattrs():
                        value = opened.__getattr__(var)
                        if not np.isscalar(value):
                            if len(value) == len(GRZICE_BANDS):
                                value = value[GRZICE_BANDS.index(self.grzice_band)]
                    else:
                        self.logger.warning(f'variable {var} not in file, filling with fillValue, {self._fillValue}')
                        value = np.ones(np.count_nonzero(good)) * self._fillValue
                else:
                    if 'ecef' in opened[var].dimensions:
                        value = opened[var][good, :]
                    elif 'bands' in opened[var].dimensions:
                        value = opened[var][i_band, good]
                    else:
                        value = opened[var][good]

                if var in self.db_vars:
                    value = 10 * np.log10(value)

                if 'time' in var:
                    value = np.vectorize(gpstime2utc_fast)(np.array(value))

                if isinstance(value, int):
                    temp_dict[var][:np.count_nonzero(good)] = np.ones(np.count_nonzero(good), dtype=int) * value
                elif isinstance(value, str):
                    temp_dict[var][:np.count_nonzero(good)] = [value for _ in range(np.count_nonzero(good))]
                elif (value.size == 1) & (np.count_nonzero(good) > 1):
                    temp_dict[var][:np.count_nonzero(good)] = np.ones(np.count_nonzero(good)) * value
                else:
                    temp_dict[var][:np.count_nonzero(good)] = np.array(value)

            temp_dict['files_read'] = local_file

            return temp_dict

    def get_good_indices(self, loaded_file, surface=None):
        """
        Get indices that are not flagged
        :param loaded_file: output of xarray.open_dataset(netcdf_file)
        :param surface: str, surface type, optional.
        :return:
        """
        if self.override_flags:
            good = np.full(len(loaded_file[self.lat_str][:]), 1, dtype=bool)
        else:
            if loaded_file.flag_refl_in_prompt_tap[GRZICE_BANDS.index(self.grzice_band)] == 1:
                good = np.full(len(loaded_file[self.lat_str][:]), 0, dtype=bool)
            else:
                good = ((loaded_file['flag_high_elevation'][:] != 1)
                        & (loaded_file['flag_low_elevation'][:] != 1)
                        & (loaded_file['flag_antenna_gain'][GRZICE_BANDS.index(self.grzice_band), :] != 1))

        if surface == 'ocean':
            good = good & (loaded_file['flag_land'][:] != 1)
        elif surface == 'land':
            good = good & (loaded_file['flag_land'][:] == 1)

        return good

    def gridstereo(self, z, lats=None, lons=None, inds=None, grid=25, hemisphere='south',
                   cmin=None, cmax=None, cmap='RdYlBu', grid_type='NSIDC', method=np.nanmean,
                   make_fig=True):
        """
        This function grids the data to passed in and returns the data grid and lons and lats.
        It passess the gridded data to pcolorstereo, which makes the figure
        can usually just use eg. gridstereo(lat, lon, data).
        I have kept the gridding and plotting separate so that mapreduce is possible.

        :param lats: M x 1 vector of decimal latitudes, will default to lats in class
        :param lons:  M x 1 vector of decimal longitudes, will default to lats in class
        :param z: M x 1 vector data to plot OR name of variable
        :param inds: vector of indices with which to index plotting data
        :param grid: grid size (km)
        :param hemisphere: 'north' or 'south'
        :param cmin: minimum for colourmap
        :param cmax: maximum for colourmap
        :param cmap: colourmap, defaults to
        :param grid: 'auto' for gridding using the extent of the points or 'NSIDC' to use the NSIDC grid sextents
        :param method: function to use for averaging over grid cell, defaults to numpy.nanmean
        :param make_fig: bool, whether to make the figure or just return the grids.
        :return: lat_grid, lon_grid, img, m
        """
        from pyproj import Proj

        if lats is None:
            lats = self.data[self.lat_str]
        if lons is None:
            lons = self.data[self.lon_str]
        if isinstance(z, str):
            z = self.data[z]

        if inds is None:
            inds = np.full(len(lats), 1, dtype=bool)

        hemisphere = hemisphere.lower()
        if hemisphere == 'antarctic':
            hemisphere = 'south'
        elif hemisphere == 'arctic':
            hemisphere = 'north'

        if len(np.unique([len(lats), len(lons), len(z)])) != 1:
            raise ValueError('lats, lons and data must all be the same size please')

        if (hemisphere != 'north') & (hemisphere != 'south'):
            raise ValueError('Hemisphere should be either "north" or "south" Default is North.')

        if hemisphere == 'north':
            hemi_mult = 1
        else:
            hemi_mult = -1

        if np.count_nonzero((z == self._fillValue) | np.isnan(z)) > 1:
            self.logger.info('removing {} {}s and nans'.format(
                np.count_nonzero((z == self._fillValue) | (np.isnan(z))), self._fillValue))
            inds = inds & (z != self._fillValue) & ~np.isnan(z)

        inds = inds & (np.sign(lats) == np.sign(hemi_mult)) & (abs(lats) > 40)

        if np.count_nonzero(lats) < 1:
            self.logger.warning(f'no remaining data after removal of nans and {self._fillValue}')
            return

        lats = lats[inds]
        lons = lons[inds]
        z = z[inds]

        if hemisphere == 'north':
            projection = Proj('epsg:3411')  # NSIDC Polar Stereographic, Arctic
        else:
            projection = Proj('epsg:3412')  # NSIDC Polar Stereographic, Antarctic

        x, y = projection(lons, lats)
        x = x / 1000
        y = y / 1000

        if grid_type == 'NSIDC':
            x_poss, y_poss = self.make_NSIDC_grids(hemisphere=hemisphere, grid=grid, out='x')
            x_temp = x_poss[0, :]
            y_temp = y_poss[:, 0]
            x_min = np.nanmin(x_temp)
            x_max = np.nanmax(x_temp) + grid
            y_min = np.nanmin(y_temp)
            y_max = np.nanmax(y_temp) + grid
        else:
            x_min = np.nanmin(x)
            x_max = np.nanmax(x)
            y_min = np.nanmin(y)
            y_max = np.nanmax(y)

            x_temp = np.arange(x_min, x_max - grid, grid)
            y_temp = np.arange(y_min, y_max - grid, grid)

            y_poss = np.transpose(np.tile(y_temp, (len(x_temp), 1)))  # km
            x_poss = np.tile(x_temp, (len(y_temp), 1))  # km

        x[(x <= min(x_poss.flatten())) | (x >= max(x_poss.flatten()) + grid)] = np.nan
        y[(y <= min(y_poss.flatten())) | (y >= max(y_poss.flatten()) + grid)] = np.nan

        Px = (np.ones(len(x)) * self._fillValue).astype(int)
        Py = (np.ones(len(y)) * self._fillValue).astype(int)
        in_bounds = ((x > min(x_poss.flatten())) & (x < max(x_poss.flatten()))
                     & (y > min(y_poss.flatten())) & (y < max(y_poss.flatten()))
                     & (z != self._fillValue))

        Px[in_bounds] = (np.floor((x[in_bounds] - min(x_poss.flatten())) / grid)).astype(int)
        Py[in_bounds] = (np.floor((y[in_bounds] - min(y_poss.flatten())) / grid)).astype(int)
        in_bounds = in_bounds & (Px < x_poss.shape[1]) & (Py < y_poss.shape[0])
        if np.count_nonzero(in_bounds) < 1:
            lat_grid = None
            lon_grid = None
            img = None
            return lat_grid, lon_grid, img

        ind = np.ravel_multi_index((Py[in_bounds], Px[in_bounds]), x_poss.shape, order='C')

        img = np.ones(len(x_poss.flatten())) * self._fillValue
        for index in np.unique(ind):
            img[index] = method(z[in_bounds][ind == index])

        img = np.reshape(img, x_poss.shape)
        img[img == BAD_VALUE] = np.nan
        lon_grid, lat_grid = projection(x_poss.flatten() * 1000, y_poss.flatten() * 1000, inverse=True)
        lon_grid = np.reshape(lon_grid, x_poss.shape)
        lat_grid = np.reshape(lat_grid, y_poss.shape)

        if make_fig:
            m = self.pcolorstereo(lat_grid, lon_grid, img, hemisphere=hemisphere, cmin=cmin, cmax=cmax, cmap=cmap)
            return lat_grid, lon_grid, img, m
        else:
            return lat_grid, lon_grid, img

    @staticmethod
    def pcolorstereo(lats, lons, data, hemisphere='north', cmin=None, cmax=None, proj=None, cmap='RdYlBu'):
        """
        Plots pre-gridded data to a map.
        :param lats: MxN grid of decimal latitudes
        :param lons: MxN grid of decimal longitudes
        :param data: MxN grid of data
        :param hemisphere: "north" or "south" also takes "antarctic" or "arctic"
        :param cmin: minimum for colourmap
        :param cmax: maximum for colourmap
        :param cmap: colourmap, defaults to RdYlBu
        :param proj: Projection to be used in the map
        :return: handle to plotted object and handle to map object.
        """
        hemisphere = hemisphere.lower()
        if hemisphere == 'antarctic':
            hemisphere = 'south'
        elif hemisphere == 'arctic':
            hemisphere = 'north'

        if ((len(np.unique([np.shape(lats)[0], np.shape(lons)[0], np.shape(data)[0]])) != 1)
                | (len(np.unique([np.shape(lats)[1], np.shape(lons)[1], np.shape(data)[1]])) != 1)):
            raise ValueError('lats, lons and data must all be the same shape please')

        if (hemisphere != 'north') & (hemisphere != 'south'):
            raise ValueError('Hemisphere should be either "north" or "south" Default is North.')

        if cmin is None:
            cmin = np.nanpercentile(data[data != BAD_VALUE], 5)
        if cmax is None:
            cmax = np.nanpercentile(data[data != BAD_VALUE], 95)

        if hemisphere == 'north':
            if proj is None:
                proj = ccrs.NorthPolarStereo(central_longitude=[-45])
            hemi_mult = 1
        else:
            if proj is None:
                proj = ccrs.SouthPolarStereo()
            hemi_mult = -1
        fig = plt.figure(figsize=(10, 10))
        m = plt.axes(projection=proj)
        m.set_extent([-180, 180, 90 * hemi_mult, 50 * hemi_mult], ccrs.PlateCarree())
        m.coastlines()
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        m.set_boundary(circle, transform=m.transAxes)
        gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.ylocator = mticker.MaxNLocator(nbins=3)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 15, 'color': 'gray'}
        gl.ylabel_style = {'size': 15, 'color': 'gray'}
        plot_data = np.array(data).astype(float)
        plot_data[plot_data == BAD_VALUE] = np.nan
        pl = m.pcolormesh(lons, lats, plot_data[0:-1, 0:-1], transform=ccrs.PlateCarree(), vmin=cmin, vmax=cmax, shading='flat',
                          cmap=cmap)

        if sum(data.flatten() == BAD_VALUE) > 0:
            np.warning('removed {} {} values - BAD_VALUE'.format(np.count_nonzero(data == BAD_VALUE), BAD_VALUE))

        plt.colorbar(pl)

        return pl, m


    def make_NSIDC_grids(self, hemisphere='south', grid=25, out='lat_lon'):
        """
        Makes the NSIDC grid
        :param hemisphere: 'north' or 'south'
        :param grid: resolution of grid, km
        :param out: 'x' if grid desired in x and y co-ordinates, 'lat_lon' if in latitude and longitude
        :return: 2D array - (x_poss, y_poss (km)) or (lat_grid, lon_grid)
        """
        x_min = self.map_reduce_x[hemisphere][0]
        x_max = self.map_reduce_x[hemisphere][1]
        y_min = self.map_reduce_y[hemisphere][0]
        y_max = self.map_reduce_y[hemisphere][1]
        x_temp = np.arange(x_min, x_max, grid)
        y_temp = np.arange(y_min, y_max, grid)

        y_poss = np.transpose(np.tile(y_temp, (len(x_temp), 1)))
        x_poss = np.tile(x_temp, (len(y_temp), 1))
        if out == 'x':
            return x_poss, y_poss   # km
        else:
            if hemisphere == 'north':
                projection = Proj('epsg:3411')  # NSIDC Polar Stereographic, Arctic
            else:
                projection = Proj('epsg:3412')  # NSIDC Polar Stereographic, Antarctic
            lon_grid, lat_grid = projection(x_poss.flatten() * 1000, y_poss.flatten() * 1000, inverse=True)
            lon_grid = np.reshape(lon_grid, x_poss.shape)
            lat_grid = np.reshape(lat_grid, x_poss.shape)

            return lat_grid, lon_grid, projection
