# 🤖 Autonomous Warehouse AMR Route Optimizer

An interactive multi-stop order-picking path planner and algorithmic benchmarking web application built with **Python**, **Streamlit**, and **Plotly**.

This project adapts classical state-space search and admissible minimum spanning tree (MST) heuristics into an applied warehouse robotics simulator—comparing **Optimal A* Search (Prim's MST Heuristic)** against a **Greedy Nearest-Item (Repeated BFS)** baseline across realistic fulfillment center layouts.

**🔗 Live Interactive App:** *(https://warehouse-amr-route-optimizer-tunlgfjfudqea8tnwvv9dp.streamlit.app/)*

---

## Key Algorithmic & Engineering Highlights
- **Admissible Multi-Stop Heuristic (`src/router.py`):** Formulates order picking over a state space of `(robot_position, unvisited_orders)` and guides A* search using an admissible lower bound combining nearest-item Manhattan distance with **Prim's Minimum Spanning Tree (MST)** weight across remaining pick locations.
- **Memoized State Evaluation:** Caches pairwise Manhattan distances and frozenset MST weights across search expansions (`md_cache` and `mst_cache`) to eliminate redundant graph computations[cite: 2].
- **Algorithmic Tradeoff Benchmarking:** Evaluates Optimal A* side-by-side against a Greedy Nearest-Item baseline (iterative Breadth-First Search) across total travel steps, search states expanded, heuristic cache hit rates, and solve latency (ms).
- **Interactive Warehouse Topologies (`app.py`):** Supports dynamic order batching across multiple facility configurations (*Standard Shelving Aisles*, *Cross-Dock Fulfillment Center*, and *High-Density U-Racks*) with side-by-side Plotly trajectory visualizations.

---

## Run Locally
```bash
git clone [https://github.com/Arrowhorse/warehouse-amr-route-optimizer.git](https://github.com/Arrowhorse/warehouse-amr-route-optimizer.git)
cd warehouse-amr-route-optimizer
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py

Attribution & Academic Acknowledgment
The underlying multi-goal state-space formulation (FoodSearchProblem / AnyFoodSearchProblem) and Prim's Minimum Spanning Tree heuristic concepts were originally inspired by the UC Berkeley CS188 Intro to AI Pacman search curriculum (created by John DeNero and Dan Klein; http://ai.berkeley.edu)[cite: 2]. All routing engine code, caching telemetry, warehouse layouts, and interactive visualization components in this repository were independently engineered as an original applied portfolio project.
