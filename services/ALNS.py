#PYTHON 
#PYTHON 
#PYTHON 
#PYTHON 
import sys
import random
import math
import numpy as np
import heapq
import time


class Initialize:
    def __init__(self, N, M, K, q, Q, d):
        self.N = N
        self.M = M
        self.K = K
        self.quantity = np.array(q, dtype=np.int32)  
        self.capacity = np.array(Q, dtype=np.int32)  
        self.distance = np.array(d, dtype=np.float32)

    def greedy(self):
        total_nodes = 1 + 2 * (self.N + self.M)
        visited = [False] * total_nodes
        solution = [[] for _ in range(self.K)]
        taxi_cost = [0.0] * self.K
        taxi_index = 0

        requests = []

        # Passenger requests
        for i in range(1, self.N + 1):
            dropoff = i + self.N + self.M
            cost = self.distance[0][i] + self.distance[i][dropoff]
            heapq.heappush(requests, (cost, 'passenger', i, dropoff))

        # Parcel requests
        for i in range(1, self.M + 1):
            pickup = i + self.N
            dropoff = i + self.M + self.N * 2
            cost = self.distance[0][pickup] + self.distance[pickup][dropoff]
            heapq.heappush(requests, (cost, 'parcel', pickup, dropoff, self.quantity[i - 1]))

        while requests:
            req = heapq.heappop(requests)

            if req[1] == 'passenger':
                _, _, pickup, dropoff = req
                if visited[pickup] or visited[dropoff]:
                    continue

                assigned = taxi_index % self.K
                taxi_index += 1

                prev = solution[assigned][-1] if solution[assigned] else 0
                solution[assigned].extend([pickup, dropoff])
                visited[pickup] = visited[dropoff] = True
                taxi_cost[assigned] += self.distance[prev][pickup] + self.distance[pickup][dropoff]

            elif req[1] == 'parcel':
                _, _, pickup, dropoff, quantity = req
                if visited[pickup] or visited[dropoff]:
                    continue

                assigned = None
                min_total_cost = float('inf')

                for i in range(self.K):
                    if quantity <= self.capacity[i]:
                        prev = solution[i][-1] if solution[i] else 0
                        cost = self.distance[prev][pickup] + self.distance[pickup][dropoff]
                        total_cost = taxi_cost[i] + cost
                        if total_cost < min_total_cost:
                            min_total_cost = total_cost
                            assigned = i

                if assigned is not None:
                    prev = solution[assigned][-1] if solution[assigned] else 0
                    solution[assigned].extend([pickup, dropoff])
                    visited[pickup] = visited[dropoff] = True
                    taxi_cost[assigned] += self.distance[prev][pickup] + self.distance[pickup][dropoff]

        for i in range(self.K):
            solution[i] = [0] + solution[i] + [0]

        return solution

class ANLS:
    def __init__(self, N, M, K, q, Q, d, solution):
        self.N = N
        self.M = M
        self.K = K
        self.quantity = np.array(q)
        self.capacity = np.array(Q)
        self.distance = np.array(d)  

        self.all_nodes = set(range(1, N + M + 1))

        self.pairs = {}
        self.pickup_to_delivery = {}
        self.delivery_to_pickup = {}

        for node in self.all_nodes:
            if 1 <= node <= N:
                dropoff = node + N + M
                self.pairs[node] = dropoff
                self.pairs[dropoff] = node
                self.pickup_to_delivery[node] = dropoff
                self.delivery_to_pickup[dropoff] = node
            elif N + 1 <= node <= N + M:
                dropoff = node + N + M
                self.pairs[node] = dropoff
                self.pairs[dropoff] = node
                self.pickup_to_delivery[node] = dropoff
                self.delivery_to_pickup[dropoff] = node

        self._route_length_cache = {}

        self.removal_operators = [
            self._worst_removal,
            self._shaw_removal,
            self._longest_route_removal,
            self._random_removal,
            self._load_balancing_removal,
        ]
        self.insertion_operators = [
            self._regret_2_insertion,
            self._greedy_insertion,
            self.random_feasibility_insertion,
            self._load_balanced_insertion,
        ]
        self.solution = solution

        self.removal_scores = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.removal_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.insertion_scores = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.insertion_weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        self.load_balance_weight = 0.3
        
        self._precompute_node_data()

    def _precompute_node_data(self):
        self.passenger_nodes = set(range(1, self.N + 1))
        self.parcel_nodes = set(range(self.N + 1, self.N + self.M + 1))
        self.delivery_nodes = set(range(self.N + self.M + 1, 2 * self.N + 2 * self.M + 1))
        
        self.node_types = {}
        for node in range(1, 2 * self.N + 2 * self.M + 1):
            if node in self.passenger_nodes:
                self.node_types[node] = 'passenger'
            elif node in self.parcel_nodes:
                self.node_types[node] = 'parcel'
            else:
                self.node_types[node] = 'delivery'

    def _route_length(self, route):
        key = tuple(route)
        cache = self._route_length_cache

        if key in cache:
            return cache[key]

        if len(route) < 2:
            cost = 0.0
        else:
            route_array = np.array(route, dtype=np.int32)
            costs = self.distance[route_array[:-1], route_array[1:]]
            cost = np.sum(costs)
        
        cache[key] = cost
        return cost

    def _cost(self, solution):
        return max(self._route_length(r) for r in solution if len(r) > 1)

    def _route_valid(self, route, capacity):
        if len(route) < 2 or route[0] != 0 or route[-1] != 0:
            return False

        load = 0
        visited = set()
        i = 1
        route_len = len(route) - 1

        while i < route_len:
            node = route[i]
            if node in visited:
                return False
            visited.add(node)

            node_type = self.node_types.get(node)
            
            if node_type == 'passenger':
                if i + 1 >= route_len or route[i + 1] != self.pairs[node]:
                    return False
                visited.add(route[i + 1])
                i += 2
            elif node_type == 'parcel':
                delivery = self.pairs[node]
                if delivery not in route[i:]:
                    return False
                load += self.quantity[node - (self.N + 1)]
                if load > capacity:
                    return False
                i += 1
            elif node_type == 'delivery':
                pickup = self.pairs[node]
                if pickup not in route[:i]:
                    return False
                load -= self.quantity[pickup - (self.N + 1)]
                if load < 0:
                    return False
                i += 1
            else:
                i += 1

        return True

    def _is_valid(self, solution):
        served_nodes = set()
        expected_size = 2 * (self.N + self.M)

        for route_id, route in enumerate(solution):
            if not self._route_valid(route, self.capacity[route_id]):
                return False

            route_nodes = set(n for n in route if n != 0)
            if not route_nodes.isdisjoint(served_nodes):
                return False
            served_nodes.update(route_nodes)

        return len(served_nodes) == expected_size

    def _remove_request_from_solution(self, solution, request_ids):
        new_solution = [route.copy() for route in solution]

        for kind, request in request_ids:
            pickup_node = request
            if 1 <= request <= self.N:
                delivery_node = request + self.N + self.M
            elif self.N + 1 <= request <= self.N + self.M:
                delivery_node = request + self.N + self.M
            else:
                raise ValueError(f"Invalid request_id: {request}")

            for route in new_solution:
                if pickup_node in route:
                    route.remove(pickup_node)
                if delivery_node in route:
                    route.remove(delivery_node)

        return new_solution

    def _load_balancing_removal(self, solution, removal_fraction=0.2):
        route_costs = [(self._route_length(r), i, r) for i, r in enumerate(solution)]
        route_costs.sort(reverse=True)
        
        avg_cost = sum(cost for cost, _, r in route_costs if len(r) > 2) / max(1, len([r for _, _, r in route_costs if len(r) > 2]))
        to_remove = []
        
        for cost, idx, route in route_costs:
            if len(route) <= 2:
                continue
                
            if cost > avg_cost * 1.2: 
                request_nodes = [(self.node_types.get(n, 'unknown'), n)
                               for n in route if n in self.pickup_to_delivery]
                
                if request_nodes:
                    num_remove = max(1, int(removal_fraction * len(request_nodes)))
                    scored_requests = []
                    for req_type, pickup in request_nodes:
                        dropoff = self.pairs[pickup]
                        pickup_idx = route.index(pickup) if pickup in route else -1
                        dropoff_idx = route.index(dropoff) if dropoff in route else -1
                        
                        if pickup_idx >= 0 and dropoff_idx >= 0:
                            contribution = 0
                            if pickup_idx > 0:
                                contribution += self.distance[route[pickup_idx-1], pickup]
                            if pickup_idx < len(route) - 1:
                                contribution += self.distance[pickup, route[pickup_idx+1]]
                            if dropoff_idx > 0 and dropoff_idx != pickup_idx + 1:
                                contribution += self.distance[route[dropoff_idx-1], dropoff]
                            if dropoff_idx < len(route) - 1:
                                contribution += self.distance[dropoff, route[dropoff_idx+1]]
                            
                            scored_requests.append((contribution, req_type, pickup))
                    
                    if scored_requests:
                        scored_requests.sort(reverse=True)
                        selected = [(t, p) for _, t, p in scored_requests[:num_remove]]
                        to_remove.extend(selected)
                        
                        if len(to_remove) >= 3:
                            break

        if not to_remove:
            return self._worst_removal(solution, removal_fraction)
            
        return self._remove_request_from_solution([r[:] for r in solution], to_remove), to_remove

    def _random_removal(self, solution, removal_fraction=0.1):
        pickups = [(self.node_types.get(n, 'unknown'), n)
                for route in solution for n in route if n in self.pickup_to_delivery]

        if not pickups:
            return [r[:] for r in solution], []

        num_remove = max(1, int(removal_fraction * len(pickups)))
        to_remove = random.sample(pickups, num_remove)
        return self._remove_request_from_solution([r[:] for r in solution], to_remove), to_remove

    def _worst_removal(self, solution, removal_fraction=0.15):
        scored = []

        for k, route in enumerate(solution):
            if len(route) <= 2:
                continue
            route_cost = self._route_length(route)
            capacity = self.capacity[k]

            pickup_nodes = [n for n in route if n in self.pickup_to_delivery]

            for pickup in pickup_nodes:
                dropoff = self.pickup_to_delivery[pickup]
                if dropoff in route:
                    new_route = [n for n in route if n not in {pickup, dropoff}]
                    if len(new_route) < 2:
                        new_route = [0, 0]
                    
                    if self._route_valid(new_route, capacity):
                        new_cost = self._route_length(new_route)
                        saving = route_cost - new_cost
                        req_type = self.node_types.get(pickup, 'unknown')
                        scored.append((saving, req_type, pickup))

        if not scored:
            return [r[:] for r in solution], []

        scored.sort(reverse=True)
        num_remove = max(1, int(removal_fraction * len(scored)))
        to_remove = [(t, p) for _, t, p in scored[:num_remove]]
        
        return self._remove_request_from_solution([r[:] for r in solution], to_remove), to_remove

    def _shaw_removal(self, solution, removal_fraction=0.2):
        served = []
        for route in solution:
            served.extend([n for n in route if n in self.pickup_to_delivery])

        if not served:
            return [r[:] for r in solution], []

        base = random.choice(served)
        rest = [n for n in served if n != base]
        if not rest:
            to_remove = [(self.node_types.get(base, 'unknown'), base)]
            return self._remove_request_from_solution([r[:] for r in solution], to_remove), to_remove

        rest_array = np.array(rest, dtype=np.int32)
        distances = self.distance[base, rest_array]
        sorted_indices = np.argsort(distances)
        
        k = max(1, int(removal_fraction * len(served)))
        max_candidates = min(len(rest), k * 2)
        candidate_indices = sorted_indices[:max_candidates]
        selected_count = min(k - 1, len(candidate_indices))
        
        if selected_count > 0:
            nearest = [rest_array[i] for i in candidate_indices[:selected_count]]
        else:
            nearest = []
            
        nodes = [base] + nearest
        to_remove = [(self.node_types.get(n, 'unknown'), n) for n in nodes]
        
        return self._remove_request_from_solution([r[:] for r in solution], to_remove), to_remove

    def _longest_route_removal(self, solution, removal_fraction=0.2):
        route_costs = [(self._route_length(r), i, r) for i, r in enumerate(solution)]
        route_costs.sort(reverse=True)
        
        num_routes_to_consider = min(2, len([r for r in route_costs if len(r[2]) > 2]))
        to_remove = []
        
        for _, idx, route in route_costs[:num_routes_to_consider]:
            request_nodes = [(self.node_types.get(n, 'unknown'), n)
                           for n in route if n in self.pickup_to_delivery]

            if request_nodes:
                num_remove = max(1, int(removal_fraction * len(request_nodes)))
                selected = random.sample(request_nodes, min(num_remove, len(request_nodes)))
                to_remove.extend(selected)
                
                if len(to_remove) >= max(1, len(request_nodes) // 3):
                    break
        
        if not to_remove:
            return [r[:] for r in solution], []
            
        return self._remove_request_from_solution([r[:] for r in solution], to_remove), to_remove

    def _generate_insertion_positions_fast(self, route, pickup, dropoff, capacity, req_type):
        valid_insertions = []
        orig_cost = self._route_length(route)
        route_len = len(route)

        if req_type == 'passenger':
            max_positions = min(route_len - 1, 8)
            positions = list(range(1, min(max_positions + 1, route_len)))
            
            for i in positions:
                new_route = route[:i] + [pickup, dropoff] + route[i:]
                if self._route_valid(new_route, capacity):
                    cost = self._route_length(new_route) - orig_cost
                    valid_insertions.append((cost, new_route))
                    if len(valid_insertions) >= 5:
                        break
        else:
            max_pickup_pos = min(4, route_len - 1)
            
            for i in range(1, max_pickup_pos + 1):
                max_dropoff_pos = min(3, route_len - i)
                
                for j in range(i + 1, min(i + max_dropoff_pos + 1, route_len + 1)):
                    new_route = route[:i] + [pickup] + route[i:j] + [dropoff] + route[j:]
                    if self._route_valid(new_route, capacity):
                        cost = self._route_length(new_route) - orig_cost
                        valid_insertions.append((cost, new_route))
                        
                        if len(valid_insertions) >= 6:
                            return valid_insertions
                            
        return valid_insertions

    def _load_balanced_insertion(self, solution, unassigned_requests, k=3):
        unassigned = unassigned_requests[:]
        max_attempts = min(len(unassigned) * 2, 50)
        attempts = 0
        
        while unassigned and attempts < max_attempts:
            attempts += 1
            route_costs = [self._route_length(route) for route in solution]
            avg_cost = sum(route_costs) / len(route_costs) if route_costs else 0

            candidates = []
            for idx, (req_type, node) in enumerate(unassigned):
                pickup = node
                dropoff = self.pairs[node]
                for k_route, route in enumerate(solution):
                    capacity = self.capacity[k_route]
                    current_route_cost = route_costs[k_route]
                    insertions = self._generate_insertion_positions_fast(route, pickup, dropoff, capacity, req_type)
                    if insertions:
                        cost_increase, new_route = min(insertions)
                        load_bonus = max(0, avg_cost - current_route_cost) * 0.5
                        adjusted_cost = cost_increase - load_bonus
                        candidates.append((adjusted_cost, k_route, new_route, idx))

            if not candidates:
                break

            candidates.sort()
            top_k = candidates[:min(len(candidates), k)]
            _, k_route, new_route, idx = random.choice(top_k)
            solution[k_route] = new_route
            unassigned.pop(idx)

        return solution, unassigned

    def _greedy_insertion(self, solution, unassigned_requests, k_routes=5, sample_size=40):
        unassigned = unassigned_requests[:]
        max_attempts = min(len(unassigned) * 2, 30)
        attempts = 0
        route_indices = list(range(self.K))

        while unassigned and attempts < max_attempts:
            attempts += 1
            candidates = []

            sampled = random.sample(unassigned, min(sample_size, len(unassigned)))

            for req_type, node in sampled:
                pickup, dropoff = node, self.pairs[node]
                sampled_routes = random.sample(route_indices, min(k_routes, self.K))

                for k in sampled_routes:
                    route = solution[k]
                    cap = self.capacity[k]
                    insertions = self._generate_insertion_positions_fast(route, pickup, dropoff, cap, req_type)
                    if insertions:
                        cost, new_route = insertions[0]
                        candidates.append((cost, k, new_route, (req_type, node)))

            if not candidates:
                break

            cost, k, new_route, selected = min(candidates)
            solution[k] = new_route
            unassigned.remove(selected)

        return solution, unassigned

    def _regret_2_insertion(self, solution, unassigned_requests, k=2):
        unassigned = unassigned_requests[:]
        max_attempts = min(len(unassigned) * 2, 60)
        attempts = 0

        while unassigned and attempts < max_attempts:
            attempts += 1
            regret_candidates = []

            for idx, (req_type, node) in enumerate(unassigned):
                pickup = node
                dropoff = self.pairs[node]
                insert_options = []

                for k_route, route in enumerate(solution):
                    capacity = self.capacity[k_route]
                    insertions = self._generate_insertion_positions_fast(route, pickup, dropoff, capacity, req_type)
                    if insertions:
                        cost, new_route = min(insertions)
                        insert_options.append((cost, k_route, new_route))

                if len(insert_options) >= 2:
                    insert_options.sort()
                    regret = insert_options[1][0] - insert_options[0][0]
                    regret_candidates.append((regret, insert_options[0][1], insert_options[0][2], idx))
                elif len(insert_options) == 1:
                    regret_candidates.append((0, insert_options[0][1], insert_options[0][2], idx))  

            if not regret_candidates:
                break

            regret_candidates.sort(reverse=True)
            top_k = regret_candidates[:min(len(regret_candidates), k)]
            _, k_route, new_route, idx = random.choice(top_k)

            solution[k_route] = new_route
            unassigned.pop(idx)

        return solution, unassigned

    def random_feasibility_insertion(self, solution, unassigned_requests):
        unassigned = unassigned_requests[:]
        random.shuffle(unassigned)
        max_attempts = min(len(unassigned) * 2, 40)
        attempts = 0

        while unassigned and attempts < max_attempts:
            attempts += 1
            idx = random.randrange(len(unassigned))
            req_type, node = unassigned[idx]
            pickup = node
            dropoff = self.pairs[node]

            route_indices = list(range(self.K))
            random.shuffle(route_indices)

            inserted = False
            for k in route_indices:
                route = solution[k]
                capacity = self.capacity[k]

                insertions = self._generate_insertion_positions_fast(route, pickup, dropoff, capacity, req_type)

                if insertions:
                    _, new_route = min(insertions)
                    solution[k] = new_route
                    unassigned.pop(idx)
                    inserted = True
                    break
                    
            if not inserted:
                continue
        
        return solution, unassigned

    def _update_operator_scores(self, removal_idx, insertion_idx, prev_max_cost, new_max_cost, decay=0.98):
        self.removal_weights *= decay
        self.insertion_weights *= decay

        improvement = prev_max_cost - new_max_cost

        if improvement > 10:  
            reward = improvement
        else:
            reward = -0.5  

        self.removal_weights[removal_idx] += reward
        self.insertion_weights[insertion_idx] += reward

    def _adaptive(self, iteration, epsilon_base=0.15, decay=0.999, min_prob=0.05):
        epsilon = max(0.02, epsilon_base * (decay ** iteration))

        def select(weights, operators):
            total_weight = np.sum(weights)
            if total_weight == 0:
                probs = np.ones(len(weights)) / len(weights)
            else:
                probs = np.maximum(weights / total_weight, min_prob)
                probs = probs / np.sum(probs)

            if random.random() < epsilon:
                return random.randint(0, len(operators) - 1)
            else:
                return np.random.choice(len(operators), p=probs)

        r_idx = select(self.removal_weights, self.removal_operators)
        i_idx = select(self.insertion_weights, self.insertion_operators)

        return (
            self.removal_operators[r_idx],
            self.insertion_operators[i_idx],
            r_idx,
            i_idx
        )
    

    def solve(self, max_iterations=100):
        random.seed(42)
        np.random.seed(42)

        is_subproblem = (self.N + self.M <= 100 or self.K <= 5)

        current_solution = [r[:] for r in self.solution]

        if is_subproblem:
            result = self._solve_core(max_iterations=max_iterations)
        else:
            result = self._local_refine_with_optimized_ALNS(current_solution)

        self.solution = result
        return result


    def _solve_core(self, max_iterations):
        current_solution = [r[:] for r in self.solution]
        best_solution = [r[:] for r in current_solution]
        best_unassigned = []
        current_unassigned = []

        def evaluate(sol, unassigned):
            route_costs = [self._route_length(r) for r in sol if len(r) > 1]
            if not route_costs:
                return 1e6
            return max(route_costs)

        current_cost = evaluate(current_solution, current_unassigned)
        best_cost = current_cost

        temperature = 100.0
        cooling_rate = 0.995

        for iteration in range(max_iterations):
            if len(self._route_length_cache) > 5000:
                self._route_length_cache.clear()

            if iteration < 10:
                removal_fraction = 0.25
                removal_op = self._load_balancing_removal
                insertion_op = self._load_balanced_insertion
                r_idx = self.removal_operators.index(removal_op)
                i_idx = self.insertion_operators.index(insertion_op)
            else:
                removal_op, insertion_op, r_idx, i_idx = self._adaptive(iteration)
                removal_fraction = 0.15

            temp_solution = [r[:] for r in current_solution]
            removed_solution, removed_requests = removal_op(temp_solution, removal_fraction)
            if not removed_requests:
                continue

            combined_requests = list(set(removed_requests + current_unassigned))
            inserted_solution, new_unassigned = insertion_op(removed_solution, combined_requests)
            if not self._is_valid(inserted_solution):
                continue

            new_cost = evaluate(inserted_solution, new_unassigned)
            cost_diff = new_cost - current_cost

            accept = False
            if new_cost < best_cost:
                best_solution = [r[:] for r in inserted_solution]
                best_unassigned = new_unassigned[:]
                best_cost = new_cost
                accept = True
            elif new_cost < current_cost:
                accept = True
            elif temperature > 1e-6 and random.random() < math.exp(-cost_diff / temperature):
                accept = True

            if accept:
                current_solution = [r[:] for r in inserted_solution]
                current_unassigned = new_unassigned[:]
                current_cost = new_cost

            self._update_operator_scores(r_idx, i_idx, current_cost, new_cost)
            temperature *= cooling_rate
        return best_solution

    def _create_clusters(self, solution, min_size=2, max_size=8):
        route_info = [
            (self._route_length(route), idx, route)
            for idx, route in enumerate(solution)
            if len(route) > 2
        ]
        np.random.shuffle(route_info)

        total_nodes = sum(len(route) - 2 for _, _, route in route_info)
        num_clusters = max(1, math.ceil(total_nodes / 120))

        estimated_size = max(min_size, min(max_size, len(route_info) // num_clusters))

        clusters = []
        current_cluster = []
        current_node_count = 0

        for cost, idx, route in route_info:
            route_nodes = len(route) - 2
            if current_node_count + route_nodes > 120:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = []
                current_node_count = 0

            current_cluster.append((cost, idx, route))
            current_node_count += route_nodes

        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def _build_subproblem_from_cluster(self, cluster, full_solution):
        sorted_cluster = sorted(cluster, key=lambda x: self._route_length(x[2]))

        selected = sorted_cluster[:2] + sorted_cluster[-2:] if len(sorted_cluster) >= 4 else sorted_cluster

        cluster_indices = [idx for _, idx, _ in selected]
        sub_solution_raw = [full_solution[i][:] for i in cluster_indices]
        sub_Q = [self.capacity[i] for i in cluster_indices]
        sub_K = len(sub_Q)

        nodes = set()
        for route in sub_solution_raw:
            for node in route:
                if node != 0:
                    nodes.add(node)

        full_nodes = set(nodes)
        for node in nodes:
            if 1 <= node <= self.N or (self.N + 1) <= node <= self.N + self.M:
                full_nodes.add(node + self.N + self.M)
            elif node > self.N + self.M:
                full_nodes.add(node - (self.N + self.M))

        full_nodes = sorted(full_nodes)
        node_map = {old: new for new, old in enumerate([0] + full_nodes)}
        reverse_map = {v: k for k, v in node_map.items()}

        size = len(node_map)
        sub_d = [[0] * size for _ in range(size)]
        for i_old, i_new in node_map.items():
            for j_old, j_new in node_map.items():
                sub_d[i_new][j_new] = self.distance[i_old][j_old]

        sub_passengers = [n for n in full_nodes if 1 <= n <= self.N]
        sub_parcels = [n for n in full_nodes if self.N + 1 <= n <= self.N + self.M]
        sub_q = [self.quantity[n - (self.N + 1)] for n in sub_parcels]
        sub_N = len(sub_passengers)
        sub_M = len(sub_parcels)

        sub_solution = []
        for route in sub_solution_raw:
            new_route = [node_map[n] for n in route if n in node_map]
            sub_solution.append(new_route)

        return sub_N, sub_M, sub_K, sub_q, sub_Q, sub_d, sub_solution, reverse_map, cluster_indices
    
    def _find_most_different_routes(self, solution):
        route_costs = [
            (self._route_length(route), idx, route)
            for idx, route in enumerate(solution)
            if len(route) > 2  
        ]

        if len(route_costs) < 2:
            return None

        route_costs.sort(key=lambda x: x[0])

        min_cost_route = route_costs[0]
        max_cost_route = route_costs[-1]

        return [min_cost_route, max_cost_route]

    def _local_refine_with_optimized_ALNS(self, solution, time_limit=260, max_no_improve_rounds=25):
        best_solution = [r[:] for r in solution]
        best_cost = self._cost(best_solution)
        start_time = time.time()
        round = 0

        no_improve_rounds = 0

        while time.time() - start_time < time_limit:
            elapsed = time.time() - start_time
            remaining = time_limit - elapsed
            if remaining <= 0:
                break

            if round % 2 == 0:
                clusters = self._create_clusters(solution)

                for cluster in clusters:
                    if time.time() - start_time > time_limit:
                        break

                    sub_N, sub_M, sub_K, sub_q, sub_Q, sub_d, sub_solution, reverse_map, cluster_indices = \
                        self._build_subproblem_from_cluster(cluster, solution)

                    sub_solver = ANLS(sub_N, sub_M, sub_K, sub_q, sub_Q, sub_d, solution=sub_solution)
                    sub_refined = sub_solver.solve(max_iterations=100)

                    for i, idx in enumerate(cluster_indices):
                        solution[idx] = [reverse_map[n] for n in sub_refined[i]]

                    current_cost = self._cost(solution)
                    if current_cost < best_cost:
                        best_solution = [r[:] for r in solution]
                        best_cost = current_cost
                        no_improve_rounds = 0
                    else:
                        no_improve_rounds += 1

            else:
                # === GLOBAL ROUTE BALANCING ===
                most_different = self._find_most_different_routes(solution)
                if most_different is not None:
                    cluster = most_different
                    sub_N, sub_M, sub_K, sub_q, sub_Q, sub_d, sub_solution, reverse_map, cluster_indices = \
                        self._build_subproblem_from_cluster(cluster, solution)

                    sub_solver = ANLS(sub_N, sub_M, sub_K, sub_q, sub_Q, sub_d, solution=sub_solution)
                    sub_refined = sub_solver.solve(max_iterations=100)

                    for i, idx in enumerate(cluster_indices):
                        solution[idx] = [reverse_map[n] for n in sub_refined[i]]

                    current_cost = self._cost(solution)
                    if current_cost < best_cost:
                        best_solution = [r[:] for r in solution]
                        best_cost = current_cost
                        no_improve_rounds = 0
                    else:
                        no_improve_rounds += 1

            round += 1

            if no_improve_rounds >= max_no_improve_rounds:
                break

        return best_solution


def main():
    input = sys.stdin.readline
    N, M, K = map(int, input().split())
    q = list(map(int, input().split()))
    Q = list(map(int, input().split()))
    d = []
    for _ in range(2 * N + 2 * M + 1):
        d.append(list(map(int, input().split())))

    initial = Initialize(N, M, K, q, Q, d)
    greedy_solution = initial.greedy()
    solver = ANLS(N, M, K, q, Q, d, solution=greedy_solution)
    best_solution = solver.solve()

    print(K)
    for route in best_solution:
        print(len(route))
        print(' '.join(map(str, route)))

if __name__ == "__main__":
    main()
