"""Published analogue data; no interpolation or battery physics lives here."""
from copy import deepcopy

ZHANG_PROFILE = {
    'id': 'zhang_2018_nmc_analogue_v1',
    'source_type': 'engineering_analogue',
    'reference_temperature_c': 25.0,
    'reference_soc': 0.5,
    'source_url': 'https://doi.org/10.3390/en11092275',
    'source_tables': 'Table 3 (capacity), Table 4 (ohmic resistance, SOC 50%)',
    'capacity_temperature_c': [-30, -20, -10, 0, 10, 25, 35, 45, 55],
    'capacity_factor': [.715, .818, .826, .873, .935, 1., 1., 1., 1.],
    'resistance_temperature_c': [-30, -20, -10, 0, 10, 25, 45, 55],
    'resistance_factor': [9.363, 6.315, 3.979, 2.507, 1.692, 1., .664, .575],
    'raw_capacity_ah': [54.6, 62.5, 63.1, 66.7, 71.4, 76.4, 79.1, 80., 80.8],
    'raw_resistance_mohm': [13.67, 9.22, 5.81, 3.66, 2.47, 1.46, .97, .84],
    'normalization': 'User-approved rounded factors; capacity capped at 1 above 25 C.',
    'out_of_range': 'stop',
    'limitations': 'Not measured Tattu data. Capacity, voltage cutoff, R(T) and Peukert may partly double-count lost availability. Requires joint calibration.',
}

AGRI_LIMITS = {
    'id': 'tattu_agri_12s_25c',
    'sku': 'TAFFAS22K1225X',
    'mass_kg': 5.85,
    'source_url': 'https://genstattu.com/tattu-agri-22000mah-44-4v-25c-12s1p-lipo-battery-pack-with-as150-f/',
    'charge_bands': [[10, 20, .5], [20, 30, 2.], [30, 60, 5.]],
    'discharge_bands': [[-20, 0, .2], [0, 10, .5], [10, 60, 5.]],
    'boundary_policy': 'At a shared boundary use the smaller adjacent limit; outside all bands zero.',
}


def temperature_defaults():
    return {'enabled': True, 'profile': deepcopy(ZHANG_PROFILE),
            'current_limit_policy': 'basic', 'agri_limits': deepcopy(AGRI_LIMITS)}
