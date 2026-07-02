
<h1 align="center">🛸 Drone MDP Solver</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Reinforcement_Learning-FF6F00?style=for-the-badge&logo=openai&logoColor=white" alt="RL" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/AI_Planning-4285F4?style=for-the-badge" alt="AI Planning" />
</p>

> An intelligent drone navigation engine modeling a stochastic grid-world as a **Markov Decision Process (MDP)**. This project implements the **Value Iteration** algorithm to compute optimal routing policies, allowing the drone to maximize cumulative rewards while surviving environmental hazards.

---

## 🌍 The Environment

The drone operates in a randomly generated $N \times M$ grid (ranging from 9x9 to 15x15). The environment is highly stochastic, meaning actions (North, South, East, West) have a probability of deviating from their intended direction based on environmental transitions.

**State Representation:** `(row, column, damage)`
*The drone has a damage capacity from 0 to 4. A damage level of 5 results in a fatal crash and terminates the episode.*

### 🛑 Terrain & Hazards
| Cell Type | Effect / Reward |
| :--- | :--- |
| **Normal Cell** | Standard movement cost |
| **Wall** | Cannot be entered |
| **Goal** | Terminal state (+300 reward) |
| **Storm Level 1** | -50 reward, +1 damage |
| **Storm Level 2** | -100 reward, +1 damage |
| **Storm Level 3** | -125 reward, +2 damage |
| **First Aid** | +25 reward, -1 damage (single use) |
| **Portal** | Teleports to paired portal, -5 reward |
| **Storm Region**| Additional expected penalty |

---

## 🧠 Core Algorithm: Value Iteration

The core objective is to extract an optimal policy ($\pi^*$) mapping every valid state to the best possible action to maximize expected cumulative reward.

1. **Initialization:** Retrieve all states and initialize baseline state values.
2. **Value Iteration:** Perform Bellman updates iteratively until the maximum value change across all states converges below a defined threshold.
3. **Policy Extraction:** Extract the optimal stochastic action for every `(row, col, damage)` configuration, returning a final policy dictionary.

---

## 📊 Analytics & Visualizations

The solver generates comprehensive visual analytics to evaluate the policy's mathematical effectiveness:

* 🌡️ **Value Function Heatmaps:** Visual representations of state values across the grid (one distinct heatmap for each damage level).
* 📉 **Convergence Curve:** Tracks the maximum value update per iteration to mathematically demonstrate the convergence of the Value Iteration process.
* 🧭 **Policy Maps:** Directional arrow overlays showing the final computed optimal action for every cell, cleanly distinguishing walls and goal states.

---

## ⚙️ Requirements & Installation

### Tech Stack
* **Core Logic:** Python, NumPy
* **Server:** Flask, Flask-CORS
* **Visualization:** Matplotlib / Seaborn

### Quick Start
1. **Clone the repository:**
   
   git clone https://github.com/amirkhedri/Drone-MDP-Solver.git
   


2. **Install dependencies:**
```bash
pip install flask flask-cors

```


3. **Run the local server:**
```bash
python server.py

```



---

## 🎓 Academic Context

This project was developed for the **Fundamentals and Applications of Artificial Intelligence** course.

* **Project:** Drone MDP Challenge


