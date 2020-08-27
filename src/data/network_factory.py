import numpy as np

from typing import List, Optional, Dict

from models.site import EVChargingSite
from models.geolocation import ReferenceGeoLocation
from algorithm.optimized_network import OptimizedEVChargingNetwork
from data.geolocation_factory import GeoLocationFactory
from shapely.geometry import Polygon


class NetworkFactory:

    def __init__(
            self,
            polygon: Polygon,
            num_sites: int,
            neighborhood_radius: int,
            gamma: float = 1.0,
            fixed_cost_mean: float = 1.0,
            fixed_cost_std: float = 0.2,
            variable_cost: float = 0.5,
            unit_revenue: float = 2.0,
            arrival_rate_mean: float = 5.0,
            arrival_rate_std: float = 1.0,
            mean_charging_time_mean: float = 1.0,
            mean_charging_time_std: float = 0.5,
            mean_max_capacity: float = 10,
            mean_min_capacity: int = 0.5,
            mean_min_supply: float = 1.0,
            budget_weight: float = 0.5
    ):
        self.polygon = polygon
        self.num_sites = num_sites
        self.neighborhood_radius = neighborhood_radius
        self.gamma = gamma
        self.fixed_cost_mean = fixed_cost_mean
        self.fixed_cost_std = fixed_cost_std
        self.variable_cost = variable_cost
        self.unit_revenue = unit_revenue
        self.arrival_rate_mean = arrival_rate_mean
        self.arrival_rate_std = arrival_rate_std
        self.mean_charging_time_mean = mean_charging_time_mean
        self.mean_charging_time_std = mean_charging_time_std
        self.mean_max_capacity = mean_max_capacity
        self.mean_min_capacity = mean_min_capacity
        self.mean_min_supply = mean_min_supply
        self.budget_weight = budget_weight

    def generate_sites(self) -> List[EVChargingSite]:

        # Generate site
        site_location = GeoLocationFactory(polygon=self.polygon).get_random_locations(num_locations=self.num_sites)

        # Generate site features that are correlated with location
        correlation_matrix = np.zeros((self.num_sites, self.num_sites))
        for i in range(self.num_sites):
            for j in range(self.num_sites + 1):
                correlation_matrix[i, j] = correlation_matrix[j, i] = np.exp(
                    - self.gamma * site_location[i].distance(site_location[j]) ** 2
                )
        arrival_rate = np.maximum(
            np.random.multivariate_normal(
                mean=self.arrival_rate_mean * np.ones(self.num_sites),
                cov=self.arrival_rate_std * correlation_matrix
            ), 0
        )
        mean_charging_time = np.maximum(
            np.random.multivariate_normal(
                mean=self.mean_charging_time_mean * np.ones(self.num_sites),
                cov=self.mean_charging_time_std * correlation_matrix
            ), 0
        )
        fixed_cost = np.maximum(
            np.random.multivariate_normal(
                mean=self.fixed_cost_mean * np.ones(self.num_sites),
                cov=self.fixed_cost_std * correlation_matrix
            ), 0
        )

        # Generate capacities
        max_capacity = np.random.poisson(lam=self.mean_max_capacity - 1, size=self.num_sites) + 1
        min_capacity = np.minimum(
            np.random.poisson(lam=self.mean_min_capacity, size=self.num_sites),
            max_capacity
        )
        return [
            EVChargingSite(
                fixed_cost=fixed_cost[i],
                variable_cost=self.variable_cost,
                unit_revenue=self.unit_revenue,
                arrival_rate=arrival_rate[i],
                mean_charging_time=mean_charging_time[i],
                max_capacity=max_capacity[i],
                min_capacity=min_capacity[i],
                geolocation=site_location[i],
                num_charging_stations=0
            ) for i in range(self.num_sites)
        ]

    def generate_reference_locations(self) -> List[ReferenceGeoLocation]:
        return [
            ReferenceGeoLocation(
                lat=location.lat,
                long=location.long,
                neighborhood_radius=self.neighborhood_radius,
                min_supply=np.random.poisson(lam=self.mean_min_supply - 1) + 1
            ) for location in GeoLocationFactory(polygon=self.polygon).get_uniform_locations_grid(
                neighborhood_radius=self.neighborhood_radius)
        ]

    def generate_network(self, params: Optional[Dict[str, any]] = None, seed: Optional[int] = None):

        # Set seed if provided
        if seed:
            np.random.seed(seed)

        # Generate sites
        sites = self.generate_sites()
        reference_locations = self.generate_reference_locations()

        # Check feasibility
        max_network_capacity = sum(site.max_capacity for site in sites)
        min_network_capacity = sum(location.min_supply for location in reference_locations)
        assert max_network_capacity > min_network_capacity, \
            "Infeasible network generated - try different seed or change parameters"

        # Return an optimized charging network instance
        return OptimizedEVChargingNetwork(
            sites=sites,
            reference_locations=reference_locations,
            budget=self.budget_weight * max_network_capacity + (1 - self.budget_weight) * min_network_capacity,
            **(params if params else {})
        )

