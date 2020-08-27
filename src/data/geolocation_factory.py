from typing import List, Optional

import numpy as np
from shapely.geometry import Polygon, Point

from models.geolocation import GeoLocation


class GeoLocationFactory:

    DEGREES_LAT_PER_KM = 1 / 110.54
    DEGREES_LONG_PER_KM_AT_EQUATOR = 1 / 111.32

    def __init__(self, polygon: Polygon):
        self.polygon = polygon

    def get_uniform_locations_grid(self, neighborhood_radius: float) -> List[GeoLocation]:
        locations = []
        min_long, min_lat, max_long, max_lat = self.polygon.bounds
        lat, long = min_lat, min_long
        while lat < max_long:
            while long < max_long:
                if self.polygon.contains(Point(long, lat)):
                    locations.append(GeoLocation(lat=lat, long=long))
                long += neighborhood_radius * GeoLocationFactory.DEGREES_LONG_PER_KM_AT_EQUATOR * np.cos(lat)
            lat = min_lat
            long += neighborhood_radius * GeoLocationFactory.DEGREES_LAT_PER_KM
        return locations

    def get_random_locations(self, num_locations: int, seed: Optional[int] = None) -> List[GeoLocation]:
        if seed:
            np.random.seed(seed)
        locations = []
        min_long, min_lat, max_long, max_lat = self.polygon.bounds
        while len(locations) < num_locations:
            lat = np.random.rand() * (max_lat - min_lat) + min_lat
            long = np.random.rand() * (max_long - min_long) + min_long
            if self.polygon.contains(Point(long, lat)):
                locations.append(GeoLocation(lat=lat, long=long))
        return locations

