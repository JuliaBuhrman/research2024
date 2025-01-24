#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import pyart
import xarray as xr
from datetime import datetime, timedelta
import glob
import metpy.calc as mpcalc
import metpy
import metpy.plots
from metpy.units import units
import cartopy.crs as ccrs
import gc
from astropy.convolution import convolve
from boto.s3.connection import S3Connection
import tempfile
import copy
import math
import eccodes
import cfgrib
import math
from math import radians, sin, cos, degrees, atan2, sqrt
import pyproj
from datetime import datetime
from datetime import timezone
import pytz
import os
import warnings
import contextlib


geodesic = pyproj.Geod(ellps='WGS84')

def find_nearest(array, value):
    '''
    Function to find index of the array in which the value is closest to

    Parameters: array (array), value (number)
    Returns: index (int)

    Example: xind = CM1calc.find_nearest(x,5)
    '''

    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx
def nearest_tobac_time(netcdf_list, tobac_lats, tobac_lons, tobac_times_datetime):
    '''
    To run this function, you must make sure you've pre-defined the generic find_nearest function first.
    netcdf_file = a single netcdf file
    morton_tobac_times_datetime = the list made from the get_storm_tobac function
    '''
    time_list = []
    for netcdf_file in netcdf_list:
        time_yoink = netcdf_file[-15:-3]
        time_yoink_dt = datetime.strptime(time_yoink, '%y%m%d%H%M%S')
        time_list.append(time_yoink_dt)
        
    time_yoink_dt_array = np.array(time_list)
    time_yoink_dt_convert = time_yoink_dt_array.astype('datetime64[s]')
    ka_time_series = pd.Series(time_yoink_dt_convert)
    
    nearest_tobac_index = []
    for time in time_yoink_dt_convert:
        tobac_index = find_nearest(tobac_times_datetime, time)
        nearest_tobac_index.append(tobac_index)
        
    tobac_times_closest = tobac_times_datetime[nearest_tobac_index]
    series_tobac_times = pd.Series(tobac_times_closest)
    series_tobac_indeces = pd.Series(nearest_tobac_index)
    
    tobac_lats_closest = tobac_lats[nearest_tobac_index]
    series_tobac_lats = pd.Series(tobac_lats_closest)
    tobac_lons_closest = tobac_lons[nearest_tobac_index]
    series_tobac_lons = pd.Series(tobac_lons_closest)

    return ka_time_series, series_tobac_times, series_tobac_indeces, series_tobac_lats, series_tobac_lons

def vehicle_correction_vad(radar,df):
    '''
    Function that creates a 'vad_corrected_velocity' field that can be used for vad calculations, 
    but should be general enough to use for stationary VADs as well as moving PPIs. 
    Other than adding the new field, the radar times are smoothly interpolated and the azimuths are 
    corrected via the GPS pandas dataframe.
    
    Parameters: pyart radar object (object), pandas dataframe of appropriate radarGPS file (dataframe)
    Returns: pyart radar object (object), speed (float), speed variance (float), bearing (float), bearing variance (float), 
             latitude (float), latitude variance (float), longitude (float), longitude variance (float)
    
    Example: radar, velmean, velvar, bearmean, bearvar, latmean, latvar, lonmean, lonvar = vehicle_correction_vad(radar,df)

    p.s. only works if the velocity is already dealiased and there is a 'corrected_velocity' field
         also only works if a single sweep is extracted, example: radar = radar.extract_sweeps([0])
    '''
    #for i in df:
    #orders the time to increase monotonically instead of having a massive step jump in the middle
    roll_mag = (np.argmax(np.abs(np.gradient(radar.time['data'])))+1)
    times = np.roll(radar.time['data'],-roll_mag) 
    
    #a complicated way to create linear increasing times (instead of steps) that start at 0 seconds after the time datum and increase to the middle of the second max time plateau (if confused, plotting it is helpful)
    #from now on, we are going to assume ray_times is the fractional seconds after the time datum the ray is gathered, and we need to roll it back to match with the rest of the data
    ray_times = np.roll(np.arange(0,((np.unique(times)[-2])/(find_nearest(times,np.unique(times)[-2])+int(np.sum(radar.time['data']==np.unique(times)[-2])/2)))*len(times)+1e-11,((np.unique(times)[-2])/(find_nearest(times,np.unique(times)[-2])+int(np.sum(radar.time['data']==np.unique(times)[-2])/2)))),roll_mag)

    radar.time['data']=ray_times
    
    #df['datetime'] = pd.to_datetime(df['ddmmyy']+df['hhmmss[UTC]'], format='%d%m%y%H%M%S')
    df['datetime'] = [datetime.strptime(d,'%d%m%y%H%M%S') for d in df['ddmmyy']+[f'{h:06}' for h in df['hhmmss[UTC]'].astype(int)]]
    beginscanindex = df.loc[df['datetime'] == datetime.strptime(radar.time['units'],'seconds since %Y-%m-%dT%H:%M:%SZ')].index
    endscanindex = df.loc[df['datetime'] == datetime.strptime(radar.time['units'],'seconds since %Y-%m-%dT%H:%M:%SZ')].index+np.ceil(np.amax(ray_times))+1
    if len(beginscanindex) == 0: # MAKES IT SO THAT IF THE DATETIME IS MISSING IT SKIPS THE FILE
        return None#, None, None, None, None, None, None, None
    dfscan = df.iloc[beginscanindex[0].astype(int):endscanindex[0].astype(int)]
    dfscan = dfscan.astype({'Bearing[degrees]': 'float'})
    dfscan = dfscan.astype({'Velocity[knots]': 'float'})
        
    ray_bearings = np.interp(ray_times,np.arange(len(dfscan)),dfscan['Bearing[degrees]'])
    ray_speeds = np.interp(ray_times,np.arange(len(dfscan)),dfscan['Velocity[knots]'])
        
    #    print('velocity [kts]',dfscan['Velocity[knots]'].mean(),'+-',dfscan['Velocity[knots]'].var())
    speed = dfscan['Velocity[knots]'].mean()
    #    print('bearing',dfscan['Bearing[degrees]'].mean(),'+-',dfscan['Bearing[degrees]'].var())
    bearing = dfscan['Bearing[degrees]'].mean()
    #    print('latitude',dfscan['Latitude'].astype(float).mean(),'+-',dfscan['Latitude'].astype(float).var())
    lat = dfscan['Latitude'].astype(float).mean()
    #    print('longitude',dfscan['Longitude'].astype(float).mean(),'+-',dfscan['Longitude'].astype(float).var())
    lon = dfscan['Longitude'].astype(float).mean()
         
    radar.azimuth['data'] += ray_bearings[:-1] #bearing
        
    rad_vel = copy.deepcopy(radar.fields['corrected_velocity'])
        
    rad_vel['data']+=(np.cos(np.deg2rad(radar.azimuth['data']-ray_bearings[:-1]))*(ray_speeds[:-1]/1.94384)*np.cos(np.deg2rad(radar.fixed_angle['data'][0])))[:,np.newaxis]
        
    #fix mask, remove points very close to radar as well as the very last bin, more often than not, = bad data
    rad_vel['data'].mask[:,:5] = True
    rad_vel['data'].mask[:,-1] = True
    radar.add_field('vad_corrected_velocity', rad_vel, replace_existing=True)
        
    return radar, dfscan['Velocity[knots]'].mean(),dfscan['Velocity[knots]'].var(),dfscan['Bearing[degrees]'].mean(),dfscan['Bearing[degrees]'].var(),dfscan['Latitude'].astype(float).mean(),dfscan['Latitude'].astype(float).var(),dfscan['Longitude'].astype(float).mean(),dfscan['Longitude'].astype(float).var()

def vads_sort(vad_netcdf_list, gps_df):
    '''
    Input:
    vad_netcdf_list: list of vads in netcdf form
    gps_df_str: the gps file path corresponding to the desired date and radar 
                (ex:'/Users/juliabman/Desktop/product_raw/ka2/GPS_ka2_20230527.txt')

    Output:
    
    '''
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    easterlies_vads = []
    westerlies_vads = []
    secondtrip_vads = []
    
    for file in vad_netcdf_list:
        
        radar = pyart.io.read(file)
        
        if np.logical_and(radar.scan_type == 'ppi', radar.fixed_angle['data'][0] > 10):
            radar = radar.extract_sweeps([0])
            #df = pd.read_csv(gps_df, dtype=str)
            radar, _, _, _, _, _, _, _, _ = vehicle_correction_vad(radar, gps_df)
            radar = radar.extract_sweeps([0])
            
            try:
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    VAD = pyart.retrieve.vad_browning(radar, 'vad_corrected_velocity', z_want=np.arange(50,2001,10),gatefilter=None) #change from 5001 -> 2001 for 2km
                    # makes vad of lowest 2km, see z_want= np.arange(50,2001,10) where 50 is the lowest bound in m and 2001 is upper
            except:
                print('not enough data')
                continue
            
            vadu = VAD.u_wind*1.94384 # meters to knots
            vadv = VAD.v_wind*1.94384
            #print(vadu)
            #print(vadv)
            V_magnitude = np.sqrt((vadu**2) + (vadv**2))
            #print(V_magnitude)
            
            if any(np.absolute(V_magnitude0) > 50 for V_magnitude0 in V_magnitude):
                #print('second trip')
                secondtrip_vads.append(file)
                continue
                
            if any(vadu_single < 0 for vadu_single in vadu):
                # if any(vadu_single < -60 for vadu_single in vadu):
                #     print('second trip')
                #     secondtrip_vads.append(file)
                    #continue
                #else:
                #print('easterlies')
                easterlies_vads.append(file)
                continue
            else:
                # if the magnitude of the wind shear between the surface and 2km is above 50 kts, prob second trip
                #if (any(np.absolute(vadu - vadu) > 50)) or (any(np.absolute(vadv - vadv)) > 50) or any(vadu_single < -60 for vadu_single in vadu) or any(vadv_single < -60 for vadv_single in vadv):
                #if any(np.absolute(V_magnitude0) > 50 for V_magnitude0 in V_magnitude):
                    #print('second trip')
                    #secondtrip_vads.append(file)
                    #continue
                #print('westerlies')
                westerlies_vads.append(file)
                continue
            
    return easterlies_vads, westerlies_vads, secondtrip_vads

def tobac_id(tobac_features_xr, storm_index):
    idx = tobac_features_xr['idx'].data
    cell = tobac_features_xr['cell'].data
    storm_indeces = np.where(cell == storm_index)
    tobac_times = tobac_features_xr['time']
    tobac_lats = np.array(tobac_features_xr['latitude'])
    tobac_lons = np.array(tobac_features_xr['longitude'])
    
    tobac_lats = tobac_lats[storm_indeces]
    tobac_lons = tobac_lons[storm_indeces]
    tobac_times = tobac_times[storm_indeces]
    cell_idx = idx[storm_indeces]
    
    tobac_lats_s = pd.Series(tobac_lats)
    tobac_lons_s = pd.Series(tobac_lons)
    tobac_times_s = pd.Series(tobac_times).astype('datetime64[s]')
    cell_idx_s = pd.Series(cell_idx)
    
    df = pd.DataFrame(pd.concat([tobac_lats_s, tobac_lons_s, 
                                       tobac_times_s, cell_idx_s], axis = 1))
    
    df.rename(columns = {0: 'latitude', 1: 'longitude', 2: 'datetime', 3: 'storm_index'}, inplace=True)
    tobac_times_datetime = tobac_times.astype('datetime64[s]')
    
    return df

def storm_speed_and_bearing(tobac_id_df, weights_list):
    storm_velocity_tobac = []
    storm_bearing_tobac = []
    storm_bearing_times = []
    latitude_weighted_tobac = []
    longitude_weighted_tobac = []
    storm_vel_weighted_tobac = []
    dist_tobac = []
    lat = tobac_id_df.latitude
    lon = tobac_id_df.longitude
    time = tobac_id_df.datetime.values.astype(float)
    weights = weights_list
    
    
    for w in range((len(lat))):
        if (w - 1 <= 0):
            print(w)
            continue
        else:
            if (w < (np.size(lat) -2)):
                print(f'{w} yuh')
                lat_over_25_minutes = np.array([lat[w-2], 
                                                        lat[w-1],
                                                        lat[w], 
                                                        lat[w+1], 
                                                        lat[w+2]])
                lon_over_25_minutes = np.array([lon[w-2], lon[w-1], lon[w], lon[w+1], lon[w+2]])
                
                storm_times = tobac_id_df.datetime.iloc[w]
                storm_bearing_times.append(storm_times)
                arraylat = np.array(lat_over_25_minutes)
                flatlat = arraylat.flatten()
                lat_weighted = np.average(flatlat, weights = weights)
                latitude_weighted_tobac.append(lat_weighted)
            
                arraylon = np.array(lon_over_25_minutes)
                flatlon = arraylon.flatten()
                lon_weighted = np.average(flatlon, weights = weights)
                longitude_weighted_tobac.append(lon_weighted)
            else:
                print('nope')
                break
                
    for i in range(len(latitude_weighted_tobac)-1):
        fwd_az, back_az, distance = geodesic.inv(longitude_weighted_tobac[i], latitude_weighted_tobac[i], longitude_weighted_tobac[i+1], latitude_weighted_tobac[i+1])
        # distance in m, az in degrees clockwise from N
        storm_bearing_tobac.append(fwd_az)
        
        for t in range(len(time)-1):
            vel = distance / (time[i+1] - time[i]) # in m/s
            storm_vel_knots = vel * 1.94384 # m/s to knots
            storm_velocity_tobac.append(storm_vel_knots)
    
        storm_bearing_array = np.array(storm_bearing_tobac).astype('int')
        storm_velocity_array = np.array(storm_velocity_tobac).astype('int')
        storm_bearing_times_array = np.array(storm_bearing_times).astype('datetime64[s]')

    return latitude_weighted_tobac, longitude_weighted_tobac, storm_bearing_array, storm_velocity_array, storm_bearing_times_array

def azshear_grib_variables(grib_nc):
    lats = pd.Series([])
    lons = pd.Series([])
    times = pd.Series([])
    shears = pd.Series([])
    lat_list = []
    lon_list = []
    time_list = []
    shear_list = []
    for i in range(len(grib_nc.time)):
        azshear = grib_nc.MergedAzShear0to2kmAGL_500mabovemeansealevel[i,:,:]
        max_shear_index = np.unravel_index(np.argmax(azshear.values), azshear.shape)
        max_shear = azshear[max_shear_index]
        time_datetime = max_shear.time.data.astype('datetime64[s]')
        lat_list.append(max_shear.latitude.data)
        lon_list.append(max_shear.longitude.data)
        time_list.append(time_datetime)
        shear_list.append(max_shear.data)
        max_lat = pd.Series(max_shear.latitude.data)
        max_lon = pd.Series(max_shear.longitude.data)
        max_time = pd.Series(time_datetime)
        max_shear_a = pd.Series(max_shear.data)
        lats = pd.concat([lats, max_lat], names = 'latitude')
        lons = pd.concat([lons, max_lon], names = 'longitude')
        times = pd.concat([times, max_time], names = 'times')
        shears = pd.concat([shears, max_shear_a], names = 'shears')

    return lats, lons, times, shears

def vad_df_new(vads_list, ka_gps, ka_time_series, tobac_times, tobac_indeces, tobac_lats, tobac_lons):
    radar_E = []
    velmean_E = []
    velvar_E = []
    bearmean_E = []
    latmean_E = []
    latvar_E = []
    lonmean_E = []
    lonvar_E = []
    for file in vads_list:
        print(file)
        try:
            ka_gps['ddmmyy'] = ka_gps['ddmmyy'].astype(str)
            ka_gps['hhmmss[UTC]'] = ka_gps['hhmmss[UTC]'].astype(str)
            read = pyart.io.read(file)
            radar = read.extract_sweeps([0])
            all_vehicle_correction = vehicle_correction_vad(radar, ka_gps)
            if all_vehicle_correction == None:
                print('broken')
                continue
            radar_E.append(all_vehicle_correction[0])
            velmean_E.append(all_vehicle_correction[1])
            velvar_E.append(all_vehicle_correction[2])
            bearmean_E.append(all_vehicle_correction[3])
            latmean_E.append(all_vehicle_correction[5])
            latvar_E.append(all_vehicle_correction[6])
            lonmean_E.append(all_vehicle_correction[7])
            lonvar_E.append(all_vehicle_correction[8])
        except KeyError:
            pass
            
    radar_column = pd.Series(np.array(radar_E))
    velmean_column = pd.Series(velmean_E)
    velvar_column = pd.Series(velvar_E)
    bearmean_column = pd.Series(bearmean_E)
    latmean_column = pd.Series(latmean_E)
    latvar_column = pd.Series(latvar_E)
    lonmean_column = pd.Series(lonmean_E)
    lonvar_column = pd.Series(lonvar_E)
    
    df = pd.DataFrame(pd.concat([pd.Series(ka_time_series),radar_column, velmean_column, velvar_column, 
                                 bearmean_column, latmean_column, latvar_column, lonmean_column, 
                                 lonvar_column, pd.Series(tobac_times), pd.Series(tobac_indeces), 
                                 pd.Series(tobac_lats), pd.Series(tobac_lons)], axis = 1))
    
    df.rename(columns={0: 'Datetime', 1: 'Radar', 2: 'Velmean', 3: 'Velvar', 4: 'Bearmean', 5: 'Latmean', 
                            6: 'Latvar', 7: 'Lonmean', 8: 'Lonvar', 9: 'tobac_times',
                            10: 'tobac_indeces', 11: 'tobac_lats', 12: 'tobac_lons', 
                           }, inplace  = True)
    return df

def polar_plot(title_string, meso_to_ka_bear, storm_bearing_tobac_vads, meso_to_ka_dist):
    # bring in the figure
    from matplotlib import rcParams, rcParamsDefault
    
    rcParams.update(rcParamsDefault)
    
    # bring in the figure
    
    #fig = plt.figure(figsize = (5,5))
    plt.clf()
    sp = plt.subplot(111,projection = 'polar')
    sp.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    sp.grid(True, linestyle = ':') # makes grid lines dotted
    sp.set_title(title_string)
    
    sp.set_theta_zero_location('E')
    sp.set_theta_direction(-1)
    # now that we have the figure set up, we can plot on it
    # r --> great circle distance from lat and lon of storm to lat and lon of vehicle
    # theta --> corrected bearing
    # in polar plots the order of plotting is (theta, r)
    
    theta_shade = np.linspace(0, np.pi/2)  # Angular range from 0° to 90°
    r_shade = np.linspace(0, 40)  # Radial range from 0 to the maximum distance
    
    # Create a meshgrid for the filled area
    sp.fill_between(theta_shade, 0, 60, color='pink', alpha=0.3, label = 'Inflow')
    
    theta = ((meso_to_ka_bear + storm_bearing_tobac_vads)% 360) * (np.pi/180)
    r = meso_to_ka_dist # - np.array(ka2_matching_velocity) - np.array(storm_velocity)
    
    print(np.shape(theta))
    print(np.shape(r))
    
    #theta2 = ka2_storm_rel_position_rad
    #r2 = distance_ka2_from_storm
    
    #ka2 = ax.scatter(theta, r, s = 10) # s changes point size
    #ka2 = ax.scatter(theta2, r2, s = 10)
    ka2  = sp.scatter(theta, r, s=10, label = 'Ka2')
    
    # Plot a vector along the 0° line
    theta_start = np.deg2rad(100)  # 0° in radians
    r_start = 0      # Start from the origin
    r_vector = 50    # Length of the vector
    
    
    # 1. Draw the radial line at 180° (pi radians)
    sp.plot([np.pi/2, np.pi/2], [0, 60], color='gray', linewidth=2)
    
    # 2. Draw the radial line at 270° (3*pi/2 radians)
    sp.plot([0,0], [0, 60], color='gray', linewidth=2, alpha = 0.7)
    
    # 3. Draw the arc from 180° to 270° at the max radius
    sp.plot(np.linspace(0, np.pi / 2, 100), [60] * 100, color='gray', linewidth=2)
    
    # Use quiver to plot the vector (dx=1 means it's along the theta=0 direction, dy=0 means no angular component)
    sp.quiver(theta_start, 0.5, 20, 0, scale_units='xy', scale=0.2, color='black', label = 'Storm Motion')
    
    plt.legend(loc = 'upper left')
    #plt.savefig('/Users/juliabman/Desktop/SLS Images/GradyKaPos')
    
    sp.plot()
    plt.show()
    
    return theta, r


def tor_id(ka_df, storm_event_data, event_type, tornado_location, timezone, daylight_savings):
    '''
    This function uses .csv from NOAA's Storm Events Database to compare 
    the times VADs were taken with when the tornado was reported.

    ka_df = dataframe you want the tor ids to be added to
    date = date the data from the storm event database was recorded (MMDDYYYY)
    event_type = is typically 'Tornado' but for posterity left it as a variable
    storm_event_data = csv from the storm event database
    tornado_location = str of the desired location name IN ALL CAPS
    timezone = str of timezone in format US/timezone (Central, Mountain, etc)
    daylight_savings = False, the database reports in standard time, even during daylight savings
    '''
    tor_begin_time = []
    tor_end_time = []
    tor_begin_date = []
    tor_end_date = []
    vads_tor_time = []
    tor_id = storm_event_data.EVENT_TYPE == event_type
    tor_loc = storm_event_data.BEGIN_LOCATION == tornado_location
    
    for i in range(len(tor_id)):
        if tor_id[i] == True and tor_loc[i] == True:
            print(storm_event_data.BEGIN_TIME[i], storm_event_data.BEGIN_LOCATION[i])
            tor_begin_time.append(storm_event_data.BEGIN_TIME[i])
            tor_end_time.append(storm_event_data.END_TIME[i])
            tor_begin_date.append(storm_event_data.BEGIN_DATE[i])
            tor_end_date.append(storm_event_data.END_DATE[i])
        else:
            continue
            
    tor_begin_time_array = np.array(tor_begin_time)
    tor_end_time_array = np.array(tor_end_time)

    tor_start_dt_list = []
    tor_end_dt_list = []
    
    for i in range(len(tor_begin_time_array)):
        tor_start = tor_begin_date[i] + str(f' {tor_begin_time_array[i]}')
        tor_start_dt = datetime.strptime(tor_start, '%m/%d/%Y %H%M')
        print(tor_start_dt)
        local_time_start = pytz.timezone(timezone).localize(tor_start_dt, is_dst = daylight_savings)
        
        if local_time_start.dst() != timedelta(0):  # DST is in effect
            # Manually adjust by adding one hour
            corrected_time_start = local_time_start + timedelta(hours=1)
        else:
            # No DST adjustment needed
            corrected_time_start = local_time_start

        start_utc = corrected_time_start.astimezone(pytz.utc)
        tor_start_dt_list.append(start_utc)
        
        tor_end = tor_end_date[i] + str(f' {tor_end_time_array[i]}')
        tor_end_dt = datetime.strptime(tor_end, '%m/%d/%Y %H%M')
        local_time_end = pytz.timezone(timezone).localize(tor_end_dt, is_dst = daylight_savings)

        if local_time_end.dst() != timedelta(0):  # DST is in effect
            # Manually adjust by adding one hour
            corrected_time_end = local_time_end + timedelta(hours=1)
        else:
            # No DST adjustment needed
            corrected_time_end = local_time_end

        end_utc = corrected_time_end.astimezone(pytz.utc)
        tor_end_dt_list.append(end_utc)

    print(tor_start_dt_list[0])
    
    for dt_str in ka_df.Datetime:
        dt_str = str(dt_str)
        dt = datetime.strptime(dt_str + str(f'+00:00'), '%Y-%m-%d %H:%M:%S%z') # makes non naive datetime
        if dt < tor_start_dt_list[0]:
            # if the vad time is <= the tor begin time AND is <= the tor begin day
            vads_tor_time.append('pre tor')
            
        if (dt >= tor_start_dt_list[0]) & (dt < tor_end_dt_list[-1]):
                vads_tor_time.append('during tor')
                
        if dt >= tor_end_dt_list[-1]:
                    vads_tor_time.append('post tor')

    #ka_df.insert(4, 'Tor', vads_tor_time)

    return vads_tor_time

def composite_u_and_v(radar_list, gps, storm_speed, storm_bearing):
    '''
    Calculates average u and v per radar object and plots them on a composite hodograph.
    radar_list = radar objects list
    storm_speed = speed of storm array with size of vads array
    storm_bearing = bearing of storm array with size of vads array
    '''
    ulist = []
    vlist = []
    for r in range(len(radar_list)):
        #print(radar)
        radar = pyart.io.read(radar_list[r])
        
        if np.logical_and(radar.scan_type == 'ppi', radar.fixed_angle['data'][0] > 10):
            radar = radar.extract_sweeps([0])
        #radar = dealias_Ka(radar)
        
            #df = pd.read_csv(gps, dtype=str)
            radar, _, _, _, _, _, _, _, _ = vehicle_correction_vad(radar, gps)
            radar = radar.extract_sweeps([0])
            #try:
            VAD = pyart.retrieve.vad_browning(radar, 'vad_corrected_velocity', z_want=np.arange(50,2001,10),gatefilter=None)
            #except:
                #print('not enough data')
                #continue
            #vadu = VAD.u_wind*1.94384 # meters to knots
            #vadv = VAD.v_wind*1.94384
            vadu = VAD.u_wind
            vadv = VAD.v_wind
            vad_height = VAD.height # heights in meters above sea level at which horizontal winds were sampled
            # change storm bearing to fix hodograph rotated angles:
            storm_bearing_hodo_rel = (storm_bearing[r] + 180) % 360
            stormu, stormv = metpy.calc.wind_components((storm_speed[r]/1.94384) * units('m/s'), storm_bearing_hodo_rel * units.deg)
            #vadu_storm_motion_corrected = vadu - (stormu.magnitude * 1.94384) # .magnitude strips the units so we dont have to give everything units
            #vadv_storm_motion_corrected = vadv - (stormv.magnitude * 1.94384)
            vadu_storm_motion_corrected = (vadu) - (stormu.magnitude)
            vadv_storm_motion_corrected = (vadv) - (stormu.magnitude)
            
            ulist.append(vadu_storm_motion_corrected)
            vlist.append(vadv_storm_motion_corrected)
            uarray = np.array(ulist)
            varray = np.array(vlist)
            udf = pd.DataFrame(uarray)
            vdf = pd.DataFrame(varray)
            udf_no_nan = udf.dropna()
            vdf_no_nan = vdf.dropna()
        
    avg_u = udf_no_nan.mean(axis=0)
    avg_v = vdf_no_nan.mean(axis=0)
    
    return avg_u, avg_v, vad_height

