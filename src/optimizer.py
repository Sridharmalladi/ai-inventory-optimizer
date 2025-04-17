from pulp import LpProblem, LpVariable, LpMinimize, lpSum
import pandas as pd

def optimize_inventory(df_optimize, total_capacity=1000):
    prob = LpProblem("Inventory_Optimization", LpMinimize)

    alloc_vars = {
        row['Product ID']: LpVariable(f"x_{row['Product ID']}", lowBound=0)
        for _, row in df_optimize.iterrows()
    }

    error_vars = {
        p: LpVariable(f"e_{p}", lowBound=0)
        for p in alloc_vars
    }

    prob += lpSum(alloc_vars[p] for p in alloc_vars) <= total_capacity

    for p in alloc_vars:
        pred = df_optimize[df_optimize['Product ID'] == p]['Predicted Demand'].values[0]
        prob += alloc_vars[p] - pred <= error_vars[p]
        prob += pred - alloc_vars[p] <= error_vars[p]

    prob += lpSum(error_vars[p] for p in error_vars)
    prob.solve()

    allocations = {p: alloc_vars[p].varValue for p in alloc_vars}
    df_optimize['Allocated Stock'] = df_optimize['Product ID'].map(allocations)
    return df_optimize