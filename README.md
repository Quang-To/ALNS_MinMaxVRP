
# Adaptive Large Neighborhood Search for Min–Max Vehicle Routing Problem

[Access the full project report](Report_OP.pdf)

---

## Abstract

This repository presents a comprehensive and extensible framework for solving the Min–Max Vehicle Routing Problem (Min–Max VRP), a challenging variant of the classical VRP that seeks to minimize the maximum route cost among all vehicles. The Min–Max VRP is of significant importance in both academic research and practical applications, especially in domains where balanced workload distribution is critical, such as ride-sharing, last-mile delivery, and fleet management. By integrating advanced metaheuristic techniques—most notably Adaptive Large Neighborhood Search (ALNS)—and providing a comparative baseline using Google OR-Tools, this project contributes to the ongoing development of robust, scalable, and fair optimization methods in transportation science.


---

## Theoretical Foundation and Scientific Motivation

The Min–Max VRP generalizes the classical VRP by introducing an equity-driven objective: minimizing the largest cost incurred by any vehicle in the fleet. This focus on fairness and operational balance addresses real-world concerns in logistics, where disproportionate workloads can lead to inefficiency, increased costs, and reduced service quality. From a scientific perspective, the Min–Max VRP poses unique challenges for combinatorial optimization, requiring algorithms that can effectively balance exploration and exploitation in a highly constrained solution space.

The ALNS algorithm implemented in this project leverages adaptive operator selection, allowing the search process to dynamically prioritize the most effective destroy and repair strategies based on historical performance. Simulated annealing-based acceptance criteria are employed to probabilistically accept non-improving solutions, thereby enhancing diversification and helping the search escape local optima. Clustering-based local refinements further improve solution quality by exploiting spatial and demand-based patterns, which is especially beneficial in instances with mixed requests (passengers and parcels). These methodological choices reflect current best practices in metaheuristic research and demonstrate the potential of adaptive approaches for equity-driven optimization problems.


---

## Methodology and Implementation Details

- **Adaptive Large Neighborhood Search (ALNS):** The ALNS framework utilizes a diverse set of destroy and repair operators, whose selection probabilities are continuously updated according to their historical contribution to solution improvement. This adaptivity ensures that the algorithm remains responsive to the evolving landscape of the search space, a key factor in solving complex VRP instances.
- **Simulated Annealing Acceptance:** By incorporating simulated annealing, the algorithm can accept worse solutions with a certain probability, which prevents premature convergence and encourages a more thorough exploration of feasible solutions. This mechanism is crucial for tackling the rugged fitness landscape typical of Min–Max VRP.
- **Clustering-based Local Search:** Local search procedures are enhanced by clustering techniques, which group requests based on spatial proximity and demand characteristics. This not only accelerates the search but also leads to more balanced and realistic route assignments, reflecting practical constraints in transportation systems.
- **Benchmarking with OR-Tools:** To provide a rigorous evaluation of the metaheuristic approach, the project includes a deterministic baseline using Google OR-Tools. Comparative analysis highlights the strengths and limitations of adaptive metaheuristics relative to state-of-the-art exact and heuristic solvers.


---

## Installation

To install all required dependencies, execute the following command in your terminal:
```bash
pip install -r requirements.txt
```


---

## Usage

The main entry point for running experiments is `main.py`, which supports both ALNS and OR-Tools solvers. The usage is as follows:
```bash
python main.py {algorithm} {input_file}
# {algorithm}: "alns" or "ortools"
# {input_file}: data/input1.txt ... data/input11.txt
```
For example, to run the ALNS solver on a specific instance:
```bash
python main.py alns data/input3.txt
```


---

## Data Structure and Experimental Design


All input instances are provided in the `data/` directory, ranging from `input1.txt` to `input11.txt`. Each file encodes a scenario of the following problem:

> Given K taxis (starting at depot 0), N passenger requests, and M parcel requests, compute routes for the taxis such that:
> - Each passenger is picked up and dropped off directly (no stops in between).
> - Each parcel has a quantity and each taxi has a parcel capacity.
> - The goal is to minimize the length of the longest route among all taxis, balancing workload.
> - Input includes the number of requests, parcel quantities, taxi capacities, and a distance matrix.
> - Output specifies the routes for each taxi, starting and ending at the depot.



**Input file format:**
- **Line 1:** N, M, K (number of passenger requests, parcel requests, and taxis)
- **Line 2:** q[1], q[2], ..., q[M] (quantity of each parcel request)
- **Line 3:** Q[1], Q[2], ..., Q[K] (parcel capacity of each taxi)
- **Line i+3 (i = 0, ..., 2N+2M):** The i-th row of the distance matrix d(i, j)

**Output file format:**
- **Line 1:** Number of taxis K
- **Line 2k (k = 1, ..., K):** Number of points in the route of taxi k (Lk)
- **Line 2k+1 (k = 1, ..., K):** Sequence of points r[0], r[1], ..., r[Lk] (route of taxi k, starting and ending at depot 0)

This structure enables comprehensive benchmarking and facilitates reproducibility of experimental results, which is essential for academic research and practical validation.


---

## Project Organization and Extensibility

The project is organized to support modular development and easy extension for future research:
```
Mini - Project/
├── main.py            # Main script for running experiments and solvers
├── services/          # Algorithm implementations
│   ├── ALNS.py        # ALNS metaheuristic
│   ├── Ortools.py     # OR-Tools baseline
│   └── __init__.py    # Module initialization
├── data/              # Input instances for benchmarking
├── requirements.txt   # Dependency specification
├── README.md          # Project documentation
├── Report_OP.pdf      # Technical report with theoretical background and results
└── .gitignore         # Version control settings
```
This structure facilitates collaborative development, integration of new algorithms, and adaptation to other VRP variants or equity-driven optimization problems.


---

## Documentation and Scientific Reporting

For a detailed exposition of the problem formulation, algorithmic design, and experimental results, refer to:
- `README.md`: Overview, installation, and usage instructions
- `Report_OP.pdf`: Full technical report, including theoretical background, methodology, and empirical analysis

