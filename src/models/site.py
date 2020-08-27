from __future__ import annotations

from math import factorial
from typing import List, Optional

import numpy as np

from models.geolocation import GeoLocation


class EVChargingSite:

    def __init__(
            self,
            fixed_cost: float,
            variable_cost: float,
            unit_revenue: float,
            arrival_rate: float,
            mean_charging_time: float,
            num_charging_stations: int,
            max_capacity: int = np.inf,
            min_capacity: int = 0,
            geolocation: Optional[GeoLocation] = None,
            name: Optional[str] = None
    ):
        self.fixed_cost = fixed_cost
        self.variable_cost = variable_cost
        self.unit_revenue = unit_revenue
        self.arrival_rate = arrival_rate
        self.effective_arrival_rate = arrival_rate
        self.mean_charging_time = mean_charging_time
        self.num_charging_stations = num_charging_stations
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity
        self.geolocation = geolocation
        self.name = name

    def expected_utilization(self) -> float:
        utilization_factor = self.effective_arrival_rate * self.mean_charging_time
        normalization_constant = 1 + sum(
            utilization_factor ** k / factorial(k)
            for k in range(1, self.num_charging_stations + 1)
        )
        return sum(
            utilization_factor ** k / factorial(k - 1)
            for k in range(1, self.num_charging_stations + 1)
        ) / normalization_constant

    def expected_percentage_utilization(self) -> float:
        return self.expected_utilization() / self.num_charging_stations * 100

    def profit(self) -> float:
        revenue = self.unit_revenue * self.expected_utilization()
        cost = self.variable_cost * self.num_charging_stations - self.fixed_cost
        return revenue - cost

    def __repr__(self) -> str:
        return str(self.__dict__)

