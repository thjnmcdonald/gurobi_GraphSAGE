# coding: utf-8
# !/usr/bin/env python

from os import chdir, getcwd
import torch
import pandas as pd
import numpy as np
import math
import gurobipy as gp
import argparse
import csv

from gurobipy import GRB
from smiles_to_molecular_graphs.single_molecule_conversion import process
from torch_geometric.utils import to_dense_adj
from optimization.nn_milp_builder_tom_version import *
from optimization.bound_tightening.bt_lp import bt_lrr as bt_lrr_ann
from graphconv.optimization.bound_tightening.bt_lp import bt_lrr as bt_lrr_conv
import graphconv.optimization.nn_milp_builder as gc_builder


parser = argparse.ArgumentParser()
# trained_models//GraphSAGE_64_3.tar

# parser.add_argument('--location', default='2022-09-16_13:45:09-GraphSAGE_64_1')
parser.add_argument('--location', default='2022-09-16_21:28:10-GraphSAGE_64_3')

parser.add_argument('--time_lim', default=1000)
parser.add_argument('--mol_len', default=4)

args = parser.parse_args()

path = str(args.location)
time_lim = float(args.time_lim)
find_mol_of_length = int(args.mol_len)

model_name = path[-14:]
gnn = model_name[0:8]
neurons = model_name[9:11]
layers = model_name[-1]

print(f'{model_name=}')
print(f'{gnn=}')
print(f'{neurons=}')
print(f'{layers=}')

n = find_mol_of_length
F = 14
d_max = 4

path_fetch = f'trained_models/{path}/{model_name}'


state_dict = torch.load(path_fetch + ".tar")
print(f'{state_dict.keys()=}')


def hard_coded(mol):
    hardcode, mol_dict = process(mol, mol_dict = True)
    edge_index = hardcode.edge_index
    adjacency_matrix = to_dense_adj(edge_index)
    hardcode_A = adjacency_matrix
    hardcode_features = hardcode.x
    rel_data = [edge_index, hardcode_A, hardcode_features]
    return rel_data, mol_dict

rel_data, mol_dict = hard_coded('CCCCl')   


def make_input_constraints(m: gp.Model, rel_data, n, F, feature_map, hard_coded = False):
    A = m.addVars(n, n, vtype=GRB.BINARY, name="A")
    x = m.addVars(n, F, vtype = GRB.BINARY, name = "x")
    m.update()

    if hard_coded:
        features = rel_data[2]
        edge_matrix = rel_data[1][0]
        feature_vectors = rel_data[-1]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    m.addConstr(A[i, j] == 1)
                else:
                    m.addConstr((A[i, j] == int(edge_matrix[i][j])))

        m.update()

        for i in range(n):
            for j in range(F):
                m.addConstr(x[i, j] == feature_vectors[i, j])

        m.update()

    else:
        num_atoms = feature_map.count('atom_type')
        num_properties = feature_map.count('properties')
        num_hybridization = feature_map.count('hybridization')
        num_neighbours = feature_map.count('neighbours')
        num_hydrogen = feature_map.count('hydrogen')
        
        features = [num_atoms, num_properties, num_hybridization, num_neighbours, num_hydrogen]        
        features_simple = [num_atoms, num_neighbours, num_hydrogen]    
        feature_cumsum = np.cumsum(features_simple)

        m.addConstr(1 == A[0,0], name = 'min constr')

        m.addConstr(1 == A[1,1], name = 'min constr')

        m.addConstr(1 == A[1,0], name = 'min constr')
        
        m.addConstrs(((A[i,i] >= A[i + 1, i + 1])
            for i in range(n-1)), name = '8a')

        m.addConstrs((A[i,i] == gp.quicksum(x[i, s] for s in range(feature_cumsum[0])) for i in range(n)), 
        name = 'one_atom')

        m.addConstrs((A[i,i] == gp.quicksum(x[i, s] for s in range(feature_cumsum[0], feature_cumsum[1])) for i in range(n)), 
        name = 'one_nbor')

        m.addConstrs((A[i,i] == gp.quicksum(x[i, s] for s in range(feature_cumsum[1], feature_cumsum[2])) for i in range(n)), 
        name = 'one_hydro')


        m.addConstrs(( (4*x[i,0] + 2*x[i,1] + 1*x[i,2] + 1*x[i,3] ==
            gp.quicksum( (s-4)*x[i, s] for s in range(feature_cumsum[0], feature_cumsum[1]))
            + gp.quicksum((t - 9) * x[i, t] for t in range(feature_cumsum[1], feature_cumsum[2]) ))
              for i in range(n)), name = '8f')

        M_4 = n + 1
        
        # big-M value, equal to amount of different feature types in feature vectors
        M_5 = len(set(feature_map))

        list_indices = list(range(n))
        sum_numbers = [[x for x in list_indices if x != i] for i in range(n)]
        
        m.addConstrs(((M_4*A[i,i] >= gp.quicksum(A[i,j] for j in sum_numbers[i]))
             for i in range(n)), name = '8g')

        # forces neighbour feature to be equal to the actual outdegree of A
        m.addConstrs(
            ( gp.quicksum(A[i,s] for s in sum_numbers[i]) == gp.quicksum((t-4)*x[i,t] for t in range(feature_cumsum[0], feature_cumsum[1])) 
                for i in range(n)), name = 'out_degree_feature'
        )

        m.addConstrs(((A[i,i] <= gp.quicksum(A[i,j] for j in sum_numbers[i]))
             for i in range(n)), name = '8h')

        m.addConstrs((A[i,j] == A[j,i] for i in range(n) for j in range(n)), name='symmetry_breaker')
        
        m.addConstrs((A[i,i]*M_5 >= gp.quicksum(x[i,s] for s in range(len(feature_map))) for i in range(n)), name = 'close_feature')

        # we need to have connected graph
        # since A[0,0] and A[1,1] are one, we only need this for i >= 1
        sum_numbers_2 = [[x for x in list_indices if x < i] for i in range(n)]
        m.addConstrs((A[i,i] <= gp.quicksum(A[i,j] for j in sum_numbers_2[i]) for i in range(2, n)), name = 'connected_graph')
        m.update()
        
    return m, A, x


def make_graphconv_milp( m, bt_procedures, x, n, F, d_max, A):
    
    #put this in another thing later

    GCN_output_len = 32

    #now we create output vars
    h_prime = m.addVars(n, GCN_output_len, lb=0, vtype=GRB.CONTINUOUS, name="h_prime")
    h_prime_vars = [[h_prime[i, j] for j in range(GCN_output_len)] for i in range(n)]
    m.update()
    
    m, H_out, bounds = gc_builder.build_graphconv_milp_and_run_bt(model = m, input_vars = x, output_vars = h_prime_vars, 
                                                        state_dict = state_dict, n = n, d_max=d_max, A = A,
                                                        bt_procedures=bt_procedures)


    return m, H_out, bounds


def make_ANN_milp(m: gp.Model, bt_procedures, n, h, rel_data=rel_data, state_dict=state_dict):
    ann_bt_procedures = [bt_lrr_ann]

    y = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='y')

    m.update()
    hi = [h[i] for i in range(len(h))]

    yi = [y]

    m = build_milp_and_run_bt(m, hi, yi, state_dict,
                              bt_procedures=ann_bt_procedures)
    m.update()

    return m, y


def make_model_optimize_and_save(bt_procedures=[bt_lrr_conv],
                                 ann_bt_procedures=[bt_lrr_ann],
                                 n=find_mol_of_length, F=F, d_max=4,
                                 rel_data=rel_data, state_dict=state_dict, bilinear=False):
    
    # create model
    m = gp.Model("GNN")
    m.Params.timeLimit = time_lim
    m.update()

    # input constraints
    m, A, x = make_input_constraints(m, rel_data, n, F, mol_dict)
    m.update()

    # graph conv layers
    m, H_out, bounds = make_graphconv_milp(m, bt_procedures, x, n, F, d_max, A)
    print(bounds)
    x_ann_zero = m.addVars(32, lb = 0, vtype=GRB.CONTINUOUS, name = 'x_ann_zero')
    m.update()

    # pooling layers
    m.addConstrs((gp.quicksum(H_out[i][j] for i in range(n)) == x_ann_zero[j] for j in range(32)), name = 'pooling')
    
    for j, (lb, ub) in enumerate(bounds[-1]):
            x_ann_zero[j].setAttr(gp.GRB.Attr.LB, (0)) #lower bound is always zero after pooling b/c ReLU. 
            x_ann_zero[j].setAttr(gp.GRB.Attr.UB, (n * ub))
            j += 1
        
    m.update()

    # ann layers 
    m, y = make_ANN_milp(m, ann_bt_procedures, n, x_ann_zero, rel_data, state_dict)
    m.update()

    # set objective, optimise and save
    objective = m.getVarByName('y')
    m.setObjective(objective, GRB.MAXIMIZE)
    m.update()
    
    
    # save the linear programming formulation
    file_name = f'{model_name}_mol_len_{find_mol_of_length}'
    m.write(f'GraphSAGE_lps/{file_name}.lp')

    # optimise and get objective
    m.optimize()
    obj = m.getObjective()
    val = obj.getValue()
    
    # save solution, relevant information and temrinal output (as backup)
    m.write(f'GraphSAGE_solutions/{file_name}.sol')

    with open(f'GraphSAGE_results.csv', mode = 'a+', newline='') as result_file:
        wr = csv.writer(result_file, quoting=csv.QUOTE_ALL)
        wr.writerow([path, gnn, layers, neurons, find_mol_of_length, m.Runtime, m.MIPGap, val, m.NumVars, m.NumConstrs, m.NumQConstrs])



    return val


def main():
    make_model_optimize_and_save()


if __name__ == "__main__":
    main()
