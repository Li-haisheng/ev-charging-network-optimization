from __future__ import annotations

from typing import List

from haversine import haversine


class GeoLocation:

    def __init__(
            self,
            lat: float,
            long: float
    ):
        self.lat = lat
        self.long = long

    def validate(self) -> None:
        assert -90 <= self.lat <= 90, f"Latitude {self.long} is outside the valid range of [-90, 90]"
        assert -180 <= self.lat <= 180, f"Longitude {self.long} is outside the valid range of [-180, 180]"

    def distance(self, other: GeoLocation) -> float:
        return haversine((self.lat, self.long), (other.lat, other.long))

    def __repr__(self) -> str:
        return f"{self.__class__}: {self.__dict__}"


class ReferenceGeoLocation(GeoLocation):

    def __init__(
            self,
            lat: float,
            long: float,
            neighborhood_radius: float,
            min_supply: int
    ):
        super().__init__(lat=lat, long=long)
        self.neighborhood_radius = neighborhood_radius
        self.min_supply = min_supply
        self.neighborhood_sites = None

    def update_neighborhood_sites(self, sites: List[GeoLocation]) -> None:
        self.neighborhood_sites = set()
        for i, site in enumerate(sites):
            if self.distance(site) < self.neighborhood_radius:
                self.neighborhood_sites.add(i)
