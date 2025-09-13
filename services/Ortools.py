from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import sys

class Solver:
    def __init__(self, N, M, K, q, Q, d):
        self.passengers = N
        self.parcels = M
        self.taxis = K
        self.quantity = q
        self.capacity = Q
        self.distance = d
        self.data = {}
        self.manager = None
        self.routing = None
        self.search_params = None

    def _data_prepocessing(self):
        self.data["num_vehicles"] = self.taxis
        self.data["depot"] = 0
        self.data["passenger"] = []
        self.data["parcels"] = []
        self.data["distance_matrix"] = self.distance
        self.data["demands"] = [0] * (2 * self.passengers + 2 * self.parcels + 1)
        self.data["vehicle_capacities"] = [0] * self.taxis

        for i in range(1, self.passengers + 1):
            pickup = i
            drop = i + self.passengers + self.parcels
            self.data["passenger"].append([pickup, drop])

        for i in range(1, self.parcels + 1):
            pickup = i + self.passengers
            drop = i + 2 * self.passengers + self.parcels
            self.data["parcels"].append([pickup, drop])
            self.data["demands"][pickup] = self.quantity[i - 1]
            self.data["demands"][drop] = -self.quantity[i - 1]

        self.data["vehicle_capacities"] = self.capacity
        self.data["quantity"] = self.quantity
        self.data["capacity"] = self.capacity

    def _solver(self):
        manager = pywrapcp.RoutingIndexManager(len(self.data["distance_matrix"]), self.data["num_vehicles"], self.data["depot"])
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return self.data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,
            1000,
            True,
            dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return self.data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            self.data["vehicle_capacities"],
            True,
            "Capacity"
        )

        for request in self.data["passenger"]:
            pickup_index = manager.NodeToIndex(request[0])
            drop_index = manager.NodeToIndex(request[1])
            routing.AddPickupAndDelivery(pickup_index, drop_index)
            routing.solver().Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(drop_index))
            routing.solver().Add(distance_dimension.CumulVar(pickup_index) <= distance_dimension.CumulVar(drop_index))
            routing.solver().Add(routing.NextVar(pickup_index) == drop_index)

        for request in self.data["parcels"]:
            pickup_index = manager.NodeToIndex(request[0])
            drop_index = manager.NodeToIndex(request[1])
            routing.AddPickupAndDelivery(pickup_index, drop_index)
            routing.solver().Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(drop_index))
            routing.solver().Add(distance_dimension.CumulVar(pickup_index) <= distance_dimension.CumulVar(drop_index))

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_params.time_limit.seconds = 290

        solution = routing.SolveWithParameters(search_params)
        if solution:
            result_routes = []
            for vehicle_id in range(self.data["num_vehicles"]):
                index = routing.Start(vehicle_id)
                route = []
                route_distance = 0
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    route.append(node_index)
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
                node_index = manager.IndexToNode(index)
                route.append(node_index)

                if len(route) > 2:
                    result_routes.append((route_distance, route))

            print(len(result_routes))
            for route_distance, route in result_routes:
                print(len(route))
                print(" ".join(map(str, route)))
        else:
            print(0)


    def solve(self):
        self._data_prepocessing()
        self._solver()

def main():
    input = sys.stdin.readline
    N, M, K = map(int, input().split())
    q = list(map(int, input().split()))
    Q = list(map(int, input().split()))
    d = [list(map(int, input().split())) for _ in range(2*N + 2*M + 1)]
    solver = Solver(N, M, K, q, Q, d)
    solver.solve()