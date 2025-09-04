import pandas as pd
import numpy as np
import os
import time

from layer_formation_analysis import run_until_first_layer

def get_g_c_from_epsilon(csv_filename, epsilon, eps_col = "epsilon", g_col = "g"):
    df = pd.read_csv(csv_filename)

    eps_vals = np.array(sorted(df[eps_col].unique()))
    eps_mask = np.isclose(df[eps_col].astype(float).values, epsilon, atol = 1e-6)
    sub = df.loc[eps_mask, [eps_col, g_col]].copy()

    g_c = float(sub[g_col].min())
    return g_c
    

def append_higher_layers(
        g_values,
        phase_csv, base_params = {},
        threshold_list = [2.0],
        eps_col = "epsilon", g_col = "g",
        layer_col = "layer", T = 10000,
        method = "LSODA", init_type = "gaussian",
        save_csv = None):
    
    df = pd.read_csv(phase_csv).copy()
    
    df[layer_col] = 2
    epsilon_values = df[eps_col].values.astype(float)
    g_values = np.asarray(g_values, dtype = float)
    results = []
    for eps in epsilon_values:
        # get critical growth rate
        g_c = get_g_c_from_epsilon(phase_csv, eps)
        start_idx = int(np.searchsorted(g_values, g_c, side = "right"))
        if start_idx >= len(g_values):
            continue

        next_thr_idx = 0
        for j in range(start_idx, len(g_values)):
            if next_thr_idx >= len(threshold_list):
                break
            
            g_val = g_values[j]
            params = {**base_params, "epsilon": eps, "g": g_val}
            t_event, h_center, _, _ = run_until_first_layer(
            params, T = T, method=method, init_type=init_type)

            if h_center >= threshold_list[next_thr_idx]:
                thr = threshold_list[next_thr_idx]
                results.append({
                    eps_col: eps,
                    g_col: g_val,
                    layer_col: int(thr)+1
                })
                next_thr_idx += 1

    out_df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)

    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok = True)
        out_df.to_csv(save_csv, index = False)




    

if __name__ == "__main__":
    csv_filename = "Results/data/phase_transition_g_eps.csv"
    print(int(1.5))
