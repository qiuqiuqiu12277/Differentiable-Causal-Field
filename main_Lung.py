import sys
import os
import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils

from copy import deepcopy

from utils import Helper
import pydot

import datetime
from gemini_utils import generate


meta_ori = pd.read_csv("Lung.csv", index_col=0)

meta = meta_ori.iloc[:,:-3].copy()
meta.index = range(len(meta))
N_sample = len(meta)
meta_idx = meta_ori.index
meta['Review'] = ''
meta['ImagePath'] = ''
for i in range(len(meta)):
    idx = meta_idx[i]
    review = meta_ori.iloc[i]['Demographics'] + ' ' + meta_ori.iloc[i]['History']
    meta.at[i, 'Review'] = review
    img_path = f"./Lung/{idx}.jpg"
    meta.at[i, 'ImagePath'] = img_path

values = meta_ori.values[:,:-3].astype(float)
names = list(meta_ori.columns[:-3])

target_name = 'score'

def iter1(helper, datetime_str):
    
    print(f"Running contrastive factor analysis")
    factors_iter1, data_interface, pairs = helper.run_contrastive_factor_analysis(meta, iter=1)
    
    data_interface['score'] = meta['score'].values.astype(float)

    V = set([target_name])
    all_factors = list(data_interface.columns)
    annotated_name = [target_name]+list(V.difference([target_name])) + list(set(all_factors).difference(V).difference([target_name]))
    annoted_values = data_interface[annotated_name].values

    V_indeces = []
    for node_index in range(1, len(annotated_name)):
        V_indeces.append(node_index)

    annotated_name = [target_name] + [annotated_name[i] for i in V_indeces]

    V = set(annotated_name)

    # Causal Discovery (FCI)
    g, edges = fci(annoted_values, alpha = 0.05, independence_test_method='kci', verbose=False)
    print("FCI causal discovery completed")

    # visualization
    pdy = GraphUtils.to_pydot(g, labels=annotated_name)
    pdy_str = pdy.to_string()
    print(pdy_str)

    (graph,) = pydot.graph_from_dot_data(pdy_str)
    
    png_path = os.path.join(output_path, f'causal_graph_iter1_{datetime_str}.png')
    
    graph.write_png(png_path)
    print(f"Causal graph visualization saved: {png_path}")
    
    print("Iteration 1 completed successfully")

    return factors_iter1, annotated_name, data_interface, pairs, g, pdy


def refine(factors_iter1, annotated_name, data_interface, pairs, g, pdy, iter=2, meta=None):
    uncertain_factors = helper.get_uncertain_factors(annotated_name, g)
    counterfactual_prompt, counterfactual_img_paths = helper.generate_causal_refinement_prompt(annotated_name, uncertain_factors, data_interface, meta, pairs, pdy=pdy)
    cf_pd = helper.generate_causal_refinement_response(counterfactual_prompt, counterfactual_img_paths, annotated_name, meta, iter=1, id_start=0)
    cf_pd.index = list(cf_pd["sample_id"])

    validated_cf_pd_causal = helper.check_counterfactual_generation(cf_pd, meta, data_interface, annotated_name, g)

    if not validated_cf_pd_causal.empty:
        meta_iter_r = pd.concat([
            meta,
            validated_cf_pd_causal[['score', 'Review', 'Image']].rename(columns={'Image': 'ImagePath'})
        ], ignore_index=True)
        meta_iter_r.index = list(meta.index) + list(validated_cf_pd_causal["sample_id"])
        print(f"Added {len(validated_cf_pd_causal)} valid counterfactual samples")
    else:
        print("No valid counterfactuals found")
        meta_iter_r = deepcopy(meta)

    used_factors = [f for f in factors_iter1 if f['name'] in annotated_name]
    factors_iter_r, data_interface_iter_r_new, pairs_iter_r = helper.run_contrastive_factor_analysis(meta_iter_r, used_factors=used_factors, iter=iter)

    data_interface_iter_r = data_interface_iter_r_new
    for i in data_interface_iter_r.index:
        if i < 1000:
            data_interface_iter_r.loc[i, 'score'] = meta.loc[i, 'score'].astype(float)
        else:
            data_interface_iter_r.loc[i, 'score'] = cf_pd.loc[i, 'score'].astype(float)
        
    all_factors_iter_r = list(data_interface_iter_r.columns)
    V = set([target_name])

    annotated_name_iter_r = [target_name]+list(V.difference([target_name])) + list(set(all_factors_iter_r).difference(V).difference([target_name]))
    annoted_values_iter_r = data_interface_iter_r[annotated_name_iter_r].values

    g_iter_r, edges_iter_r = fci(annoted_values_iter_r, alpha = 0.05, independence_test_method='kci', verbose=False)
    
    pdy_iter_r = GraphUtils.to_pydot(g_iter_r, labels=annotated_name_iter_r)
    pdy_str_iter_r = pdy_iter_r.to_string()
    print(pdy_str_iter_r)
    
    (graph_iter_r,) = pydot.graph_from_dot_data(pdy_str_iter_r)
    
    png_path = os.path.join(output_path, f'causal_graph_iter{iter}_{datetime_str}.png')
    
    graph_iter_r.write_png(png_path)
    print(f"Causal graph visualization saved: {png_path}")

    return factors_iter_r, annotated_name_iter_r, data_interface_iter_r, pairs_iter_r, g_iter_r, pdy_iter_r, meta_iter_r



if __name__ == "__main__":
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = "./outputs/Lung4"
    os.makedirs(output_path, exist_ok=True)
    
    helper = Helper(datetime_str=datetime_str, N_sample=N_sample, generate=generate, output_path=output_path, dataset="Lung4")

    factors_iter1, annotated_name, data_interface, pairs_iter1, g, pdy = iter1(helper=helper, datetime_str=datetime_str)

    pdy_str = pdy.to_string()

    factors_iter_r, annotated_name_iter_r, data_interface_iter_r, pairs_iter_r, g_iter_r, pdy_iter_r, meta_iter_r = refine(factors_iter1, annotated_name, data_interface, pairs_iter1, g, pdy, iter=2, meta=meta)

    pdy_str_iter_r = pdy_iter_r.to_string()

        
            