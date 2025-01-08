#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:47:30 2023

@author: chidesiv
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:41:29 2023

@author: chidesiv
"""

# import cdsapi

# c = cdsapi.Client()

# c.retrieve(
#     'reanalysis-era5-single-levels-monthly-means',
#     {
#         'format': 'netcdf',
#         'product_type': 'monthly_averaged_reanalysis',
#         'variable': [
#             '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
#             '2m_temperature', 'evaporation', 'mean_sea_level_pressure',
#             'skin_temperature', 'surface_net_solar_radiation', 'total_precipitation',
#         ],
#         'year': [
#             str(year) for year in range(1940, 2023)
#         ],
#         'area': [
#             55, -5, 40,
#             10,
#         ],
#         'time': '00:00',
#         'month': [
#             '01', '02', '03',
#             '04', '05', '06',
#             '07', '08', '09',
#             '10', '11', '12',
#         ],
#     },
#     'ERA5_monthly_1940_2022.nc')


import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': [
            'mean_total_precipitation_rate', '2m_temperature', '10m_wind_speed', 'surface_net_solar_radiation'
        ],
        'year': [
            str(year) for year in range(1940, 2023)
        ],
        'area': [
            55, -5, 40,
            10,
        ],
        'time': '00:00',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
    },
    'ERA5_monthly_1940_2023_new.nc')