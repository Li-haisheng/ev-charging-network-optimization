from __future__ import annotations

from typing import List

import numpy as np

from models.site import EVChargingSite


class EVChargingNetwork:

    def __init__(
            self,
            sites: List[EVChargingSite],
            gamma: float
    ):
        self.sites = sites
        self.gamma = gamma
        self._compute_distance_matrix()
        self.update_effective_arrival_rates()

    def _compute_distance_matrix(self) -> None:
        num_sites = len(self.sites)
        self.distance_matrix = np.zeros((num_sites, num_sites))
        for i in range(num_sites):
            for j in range(i + 1, num_sites):
                self.distance_matrix[i, j] = self.distance_matrix[j, i] = self.sites[i].geolocation.distance(
                    self.sites[j].geolocation)

    def update_effective_arrival_rates(self) -> None:
        for i, site in enumerate(self.sites):
            if site.num_charging_stations:
                site.effective_arrival_rate = site.arrival_rate / (
                        np.exp(-self.gamma * self.distance_matrix[i] ** 2)
                        * np.array([other.num_charging_stations / site.num_charging_stations for other in self.sites])
                ).sum()

    def total_charging_stations(self) -> int:
        return sum(site.num_charging_stations for site in self.sites)

    def total_expected_utilization(self) -> float:
        return sum(site.expected_utilization() for site in self.sites)

    def total_expected_percentage_utilization(self) -> float:
        return self.total_expected_utilization() / self.total_charging_stations() * 100

    def total_profit(self) -> float:
        return sum(site.profit() for site in self.sites)

    def __repr__(self) -> str:
        return f"{self.__class__}: {self.__dict__}"
