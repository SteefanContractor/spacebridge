import numpy as np
import pandas as pd

def norm(ser, valid_range):
    """
    Normalize given series, ser, based on the valid range
    """
    min = valid_range[0]
    max = valid_range[1]
    return (ser - min)/(max - min)

def get_series(xr_gnss, varname, use_master_flag=True, normalize=True):
    """
    xr_gnss: an xarray dataset or dataArray 
    varname: name of the variable to be extracted
    use_master_flag: Boolean indicating whether to 
                    include only the timesteps where
                    flag_master == 0
                    
    Returns: a numpy array of values of given variable
            name split into multiple columns if variable
            has bands or ecef dimensions.
    """
    split_by_bands = ['reflectivity',
            'snr_reflected',
            'snr_direct',
            'phase_noise',
            'excess_phase_noise',
            'power_reflected',
            'power_direct',
            'antenna_gain_reflected',
            'antenna_gain_direct',
            ]
    split_by_ecef = ['rx_pos',
            'tx_pos',
            'spec_pos',]
    
    if varname in split_by_ecef:
        ser = xr_gnss[varname].T[:, xr_gnss.flag_master == 0].values if use_master_flag else xr_gnss[var].T.values
    elif varname in split_by_bands:
        ser = xr_gnss[varname][:, xr_gnss.flag_master == 0].values if use_master_flag else xr_gnss[var].values
    else: 
        ser = xr_gnss[varname][xr_gnss.flag_master == 0].values if use_master_flag else xr_gnss[var].values
    ser = norm(ser, valid_range=xr_gnss[varname].valid_range) if normalize else ser

    return ser
  
  
def extract_variables(xr_gnss, use_master_flag=True, normalize=True):
    """
    xr_gnss: an xarray dataset or dataarray
    use_master_flag: Boolean indicating whether to 
                    include only the timesteps where
                    flag_master == 0
    
    Returns: a numpy array containing the rectangular 
            data with variables as columns and timesteps as 
            rows, and a vector of column names.
    """
    keys = ['reflectivity',
            'snr_reflected',
            'snr_direct',
            'phase_noise',
            'excess_phase_noise',
            'power_reflected',
            'power_direct',
            'antenna_gain_reflected',
            'antenna_gain_direct',
            'rx_pos',
            'tx_pos',
            'spec_pos',
            'angle_of_elevation',
            'longitude',
            'latitude']
    
    data = []
    columns = []
    bands = xr_gnss.bands.values
    ecef = ['x','y','z']
    if any(xr_gnss.flag_master == 0):
        dt = pd.to_datetime(f'{xr_gnss.year}-{xr_gnss.month:02d}-{xr_gnss.day:02d} {xr_gnss.hour:02d}:{xr_gnss.minute:02d}:{xr_gnss.second}')
        datetimes = pd.to_datetime(np.append(np.array([np.datetime64(dt)]), np.datetime64(dt) + np.cumsum(np.diff(xr_gnss.time[xr_gnss.flag_master == 0].values))))
        dates = datetimes.strftime("%Y%m%d")
        times = datetimes.strftime("%H%M%S.%f")
        data.append(dates)
        columns.append('date')
        data.append(times)
        columns.append('time')
        for var in keys:
            vardata = get_series(xr_gnss, var, use_master_flag=use_master_flag, normalize=normalize)
            
            if len(vardata.shape) == 2:
                numser = vardata.shape[0]
                [data.append(ser) for ser in vardata]
                suffix = [bands[num] if numser == 2 else ecef[num] if numser == 3 else None for num in range(numser)]
                [columns.append(var + str(suf)) for suf in suffix]
            else:
                data.append(vardata)
                columns.append(var)

    
    return np.array(data).T, columns
    
    

def latlon_to_xyidx(lon, lat, dslon, dslat):
    # First, find the index of the grid point nearest a specific lat/lon.   
    abslat = (dslat-lat)**2
    abslon = (dslon-lon)**2
    c = np.sqrt(abslon + abslat)

    ([xloc], [yloc]) = np.where(c == np.min(c))
    return xloc, yloc
  
def create_label_columns(df, dsit):
    """
    df: Pandas dataFrame with all feature variables
    dsit: icetype netcdf dataset
    """
    lons = df.longitude.values
    lats = df.latitude.values
    
    xy = [latlon_to_xyidx(lon, lat,dsit.LON.values, dsit.LAT.values) for (lon, lat) in zip(lons, lats)]
    df['YI_conc'] = np.array([dsit.YI.sel(X=x,Y=y).values for (x,y) in xy])
    df['FYI_conc'] = np.array([dsit.FYI.sel(X=x,Y=y).values for (x,y) in xy])
    df['MYI_conc'] = np.array([dsit.MYI.sel(X=x,Y=y).values for (x,y) in xy])
    
    return df