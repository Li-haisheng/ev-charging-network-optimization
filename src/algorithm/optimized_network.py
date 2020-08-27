
from copy import deepcopy
from typing import List, Optional

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from models.geolocation import ReferenceGeoLocation
from models.network import EVChargingNetwork
from models.site import EVChargingSite

import logging

logger = logging.getLogger()


class OptimizedEVChargingNetwork(EVChargingNetwork):

    def __init__(
            self,
            sites: List[EVChargingSite],
            reference_locations: List[ReferenceGeoLocation],
            budget: int,
            gamma: float = 1.0,
            epsilon: float = 5e-2,
            sampling_temperature: float = 1.0,
            num_local_search_iters: int = 100,
            mip_gap: Optional[float] = None,
            time_limit: Optional[float] = None,
            verbose: bool = False
    ):
        super().__init__(sites=sites, gamma=gamma)
        self.reference_locations = reference_locations
        self.budget = budget
        self.epsilon = epsilon
        self.sampling_temperature = sampling_temperature
        self.num_local_search_iters = num_local_search_iters
        self.mip_gap = mip_gap
        self.time_limit = time_limit
        self.verbose = verbose
        self.local_search_history = []
        self._update_neighborhood_sites()

    def _update_neighborhood_sites(self) -> None:
        for location in self.reference_locations:
            location.update_neighborhood_sites(sites=self.sites)

    def _is_feasible(self) -> bool:
        return (
                self.total_charging_stations() <= self.budget
                & np.all(site.num_charging_stations <= site.max_capaicty for site in self.sites)
                & np.all(site.num_charging_stations >= site.min_capaicty for site in self.sites)
                & np.all(
            sum(self.sites[i].num_charging_stations for i in location.neighborhood_sites) >= location.min_supply
            for location in self.reference_locations
        )
                & np.all(isinstance(site.num_charging_stations, int) for site in self.sites)
        )

    def _solve_relaxation(self) -> None:

        # Calculate marginal profits
        num_sites = len(self.sites)
        num_stations = max(site.max_capacity for site in self.sites)
        marginal_profit = np.zeros((num_sites, num_stations))
        for i, site in enumerate(self.sites):
            site_ = deepcopy(site)
            site_.effective_arrival_rate = site.arrival_rate
            prev_profit = 0
            for j in range(site.max_capacity):
                site_.num_charging_stations += 1
                profit = site_.profit()
                marginal_profit[i, j] = profit - prev_profit
                prev_profit = profit

        # Calculate adjusted max capacities
        adjusted_max_capacity = np.zeros(num_sites)
        for i, site in enumerate(self.sites):
            adjusted_max_capacity[i] = min(
                site.max_capacity,
                np.min(self.epsilon * np.exp(self.gamma * self.distance_matrix[i] ** 2))
            )

        # Initialize model and variables
        model = gp.Model()
        z = model.addVars(num_sites, num_stations, vtype=GRB.BINARY)

        # Set constraints for linear relaxation
        model.addConstrs(z.sum() <= self.budget)
        model.addConstrs(z.sum(i, "*") <= adjusted_max_capacity[i] for i in range(num_sites))
        model.addConstrs(z.sum(i, "*") >= site.min_capacity for i, site in enumerate(self.sites))
        model.addConstrs(
            gp.quicksum(
                z[i, k] for i in location.neighborhood_sites for k in range(num_stations)
            ) >= location.min_supply for location in self.reference_locations
        )
        model.addConstrs(z[i, k] >= z[i, k + 1] for i in range(num_sites) for k in range(num_stations - 1))

        # Set objective for linear relaxation
        model.setObjective(
            gp.quicksum(marginal_profit[i, k] * z[i, k] for i in range(num_sites) for k in range(num_stations)),
            GRB.MAXIMIZE
        )

        # Update solver parameters if provided
        if self.mip_gap:
            model.params.MIPGap = self.mip_gap
        if self.time_limit:
            model.params.TimeLimit = self.time_limit
        model.params.OutputFlag = self.verbose

        # Optimize model
        model.optimize()

        # Update network based on solution to linear relaxation
        z = model.getAttr("x", z)
        for i, site in enumerate(self.sites):
            site.num_charging_stations = sum(z[i, k] for k in range(num_stations))
        self.update_effective_arrival_rates()

    def _optimize_pair(
            self,
            site: EVChargingSite,
            neighbor: EVChargingSite
    ) -> float:
        best_solution = site.num_charging_stations, neighbor.num_charging_stations
        best_obj_val = self.total_profit()
        num_available_stations = sum(best_solution)
        for z in range(site.min_capacity, site.max_capacity + 1):
            for z_ in range(neighbor.min_capacity, min(neighbor.max_capacity, num_available_stations - z) + 1):
                site.num_charging_stations, neighbor.num_charging_stations = z, z_
                if self._is_feasible():
                    self.update_effective_arrival_rates()
                    obj_val = self.total_profit()
                    if obj_val > best_obj_val:
                        best_solution = z, z_
                        best_obj_val = obj_val
        site.num_charging_stations, neighbor.num_charging_stations = best_solution
        self.update_effective_arrival_rates()
        return best_obj_val

    def _local_search(self) -> None:
        num_sites = len(self.sites)
        for t in range(self.num_local_search_iters):
            i = np.random.choice(num_sites)
            sampling_weights = self.distance_matrix[i] * self.sampling_temperature
            sampling_weights[i] = 0
            j = np.random.choice(num_sites, p=sampling_weights / sampling_weights.sum())
            best_obj_val = self._optimize_pair(
                site=self.sites[j],
                neighbor=self.sites[j]
            )
            self.local_search_history.append(best_obj_val)
            if self.verbose:
                logger.info(
                    f"Completed local search iteration {t}/{self.num_local_search_iters}"
                    f" - best objective value: {'{0:.2f}'.format(best_obj_val)}"
                )

    def solve(self) -> None:
        if self.verbose:
            logger.info("Solving linear relaxation...")
        self._solve_relaxation()
        if self.verbose:
            logger.info("Performing local search...")
        self._local_search()
