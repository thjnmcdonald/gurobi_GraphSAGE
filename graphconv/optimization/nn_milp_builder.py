"""
Created 17 January 2019
Bjarne Grimstad, bjarne.grimstad@gmail.no

Methods for building MILP models (Gurobi) of neural networks (TensorFlow/Keras models)
"""

import gurobi as gpy
import numpy as np
from graphconv.optimization.bound_tightening.bt_utils import run_bound_tightening
from graphconv.optimization.bound_tightening import *
from graphconv.optimization.bound_tightening.bt_lp import bt_lrr, bt_rr


def get_weights(state_dict):
    """
    Return weights of Keras model as a list of tuples (w, b), where w is a numpy array of weights and b is a numpy array
    of biases for a layer. The order of the list is the same as the layers in the model.
    :param model: Keras model
    :return: List of layer weights (w, b)
    """
    weight_list = []
    root_weight_list = []
    
    for i, weights in enumerate(state_dict):
        if 'layers' in weights:
            if 'lin_r' in weights:
                root_weight_list.append((np.array(state_dict[str(weights)])).transpose())
            elif 'lin_l' in weights:
                weight_list.append((np.array(state_dict[str(weights)])).transpose())

    return list(zip(weight_list, root_weight_list))


def update_bounds(x_var, s_var, z_var, lb=None, ub=None):
    """
    Update upper bound of neuron modeled as
    W x_prev + b = x - s,
    s >= 0
    x >= 0
    z in {0, 1}
    NOTE: Remember to update model after updating bounds
    :param lb: Lower bound
    :param ub: Upper bound
    :param x_var: x variable
    :param s_var: s variable
    :param z_var: z variable used in indicator constraints (z=0 => x <= 0 and z=1 => s <= 0)
    :return: None
    """
    # NOTE: default feasibility tolerance of Gurobi is 1e-6 so we cannot expect higher accuracy
    gurobi_ftol = 1e-6  # default feasibility tolerance of Gurobi
    ftol = gurobi_ftol + 1e-9  # Feasibility tolerance
    strict_ftol = 1e-12  # Strict feasibility tolerance

    if lb and ub:
        # If bounds are infeasible by a tiny amount, we remedy.
        if lb > ub and abs(ub - lb) < strict_ftol:
            diff = lb - ub
            lb -= diff/2
            ub += diff/2
        assert lb <= ub

    if lb:
        # TODO: This is very strict since constraint violation is allowed in Gurobi (will fail at some point)
        # assert x_var.getAttr(gpy.GRB.Attr.UB) + ftol >= lb >= -(s_var.getAttr(gpy.GRB.Attr.UB) + ftol)

        # Update bounds
        if lb >= 0:
            x_var.setAttr(gpy.GRB.Attr.LB, lb)
            s_var.setAttr(gpy.GRB.Attr.UB, 0)
            z_var.setAttr(gpy.GRB.Attr.LB, 1)  # Neuron activated
        else:
            s_var.setAttr(gpy.GRB.Attr.UB, -lb)

    if ub:
        # TODO: This is very strict since constraint violation is allowed in Gurobi (will fail at some point)
        # assert x_var.getAttr(gpy.GRB.Attr.UB) + ftol >= ub >= -(s_var.getAttr(gpy.GRB.Attr.UB) + ftol)

        # Update bounds
        if ub >= 0:
            x_var.setAttr(gpy.GRB.Attr.UB, ub)
        else:
            x_var.setAttr(gpy.GRB.Attr.UB, 0)
            s_var.setAttr(gpy.GRB.Attr.LB, -ub)
            z_var.setAttr(gpy.GRB.Attr.UB, 0)  # Neuron deactivated

    # Test bounds (allowing infeasibility with a strict tolerance)
    assert x_var.getAttr(gpy.GRB.Attr.UB) >= x_var.getAttr(gpy.GRB.Attr.LB) - strict_ftol
    assert s_var.getAttr(gpy.GRB.Attr.UB) >= s_var.getAttr(gpy.GRB.Attr.LB) - strict_ftol
    assert z_var.getAttr(gpy.GRB.Attr.UB) >= z_var.getAttr(gpy.GRB.Attr.LB) - strict_ftol


def graph_conv_milp_builder(model: gpy.Model, input_vars, output_vars, weights, n, A, hidden_bounds=None, c_id=str('gc')):
    """
    Builds a MILP model of a neural network with ReLU activations.

    Assuming that NN model has K+1 layers, one input layers, K-1 hidden layers with ReLU activation,
    and one linear output layer. That is, the assumed architecture is:
    input -> ReLU -> ... -> ReLU -> output.

    Variables are named as x_k_j_k, where i is the layer number, and j is the unit number (of layer i), and k is the
    constraint ID (c_id).

    :param model: MILP model to add variables and constraints (Gurobi model)
    :param input_vars: List of input variables (Gurobi variables)
    :param output_vars: Output variables (Gurobi variables)
    :param weights: NN model weights
    :param hidden_bounds: Bounds on hidden neurons in NN model
    :param c_id: Constraint ID (required to uniquely name variables when adding multiple NN models to MILP model)
    :return: model
    """
    # Hidden layers (ReLU)
    # Layers are numbered from 1 and up
    num_layers = len(weights)

    # find a better way to find M.
    M_1 = 1
    F = 14

    # we start with an input layer:
    b_init = model.addVars(n,n,F, lb=0, ub = 1, vtype = gpy.GRB.BINARY, name=f'b_init')
    model.update()

    model.addConstrs((input_vars[l][f] - M_1*(1 - A[i,l]) <= b_init[i,l,f] for i in range(n) for l in range(n) for f in range(F)), name = '24d')
    model.addConstrs((input_vars[l][f] + M_1*(1 - A[i,l]) >= b_init[i,l,f] for i in range(n) for l in range(n) for f in range(F)), name = '24e')
    model.addConstrs((-M_1*(A[i,l]) <= b_init[i,l,f] for i in range(n) for l in range(n) for f in range(F)), name = '24f-a')
    model.addConstrs((M_1*(A[i,l]) >= b_init[i,l,f] for i in range(n) for l in range(n) for f in range(F)), name = '24f-b')
    model.update()

    x_prev = input_vars

    for k, weights_k, in enumerate(weights):
        w, w_root = weights_k
        nx, nu = w.shape  # Number of inputs to layer and number of units in layer
        
        assert nx == len(x_prev[0])
        assert w.shape == w_root.shape

        # Layer variables
        x_k, s_k, z_k, b_k = [], [], [], []
        
        for i in range(n):
            x_i, s_i, z_i, b_i = [], [], [], []           
            
            for j in range(nu): #nu is the amount of features in the next row of neurons. basically the amount of rows of w^T
                # Create variables for layer

                x_j = model.addVar(lb=0, vtype=gpy.GRB.CONTINUOUS, name=f'x_{k}_{i}_{j}_{c_id}')
                s_j = model.addVar(lb=0, vtype=gpy.GRB.CONTINUOUS, name=f's_{k}_{i}_{j}_{c_id}')
                z_j = model.addVar(vtype=gpy.GRB.BINARY, name=f'z_{k}_{i}_{j}_{c_id}')
                # b_k[i][j][l] selects node b^k_{il}_{j}, so feature j of a connected node pair (i,j) in the k-th layer. 
                b_l = model.addVars(n, lb=0, vtype=gpy.GRB.CONTINUOUS, name=f'b_{k}_{i}_{j}_{c_id}') 
                
                model.update()

                x_i.append(x_j)
                s_i.append(s_j)
                z_i.append(z_j)
                b_i.append(b_l)

                # Update bounds
                if hidden_bounds[k][j]:
                    # dit moet nog aangepast worden zo. 
                    lb, ub = hidden_bounds[k][j]
                    update_bounds(x_j, s_j, z_j, lb, ub)
                    model.update()

                sum_list = [num for num in range(n) if num != i]

                # for the instance where there is only one layer. 
                if k == 0 and num_layers == 1:
                    model.addConstr(gpy.quicksum(w_root[f,j] * input_vars[i][f] for f in range(nx))
                    + gpy.quicksum(w[f,j] * gpy.quicksum(b_init[i, l, f] for l in sum_list) for f in range(nx)), 
                    gpy.GRB.EQUAL, output_vars[i][j] - s_i[j])

                elif k == 0: #k == 0 is the same is k == 1 in the milp formulation in the thesis. 
                    model.addConstr(gpy.quicksum(w_root[f,j] * input_vars[i][f] for f in range(nx))
                    + gpy.quicksum(w[f,j] * gpy.quicksum(b_init[i, l, f] for l in sum_list) for f in range(nx)), 
                    gpy.GRB.EQUAL, x_i[j] - s_i[j])

                    #x_i[j] is the same as H_{ij}^1 in the milp formulation
                    # after running for k == 0 we have H_{ij}^1 for all i and j, which we can store in x_prev, and b_prev?
                
                elif k != (num_layers - 1): #k == 1 is the same is k == 2 in the milp formulation. 
                    model.addConstr(gpy.quicksum(w_root[f,j] * x_prev[i][f] for f in range(nx))
                    + gpy.quicksum(w[f,j] * gpy.quicksum(b_prev[i][f][l] for l in sum_list) for f in range(nx)), 
                    gpy.GRB.EQUAL, x_i[j] - s_i[j])

                    #dit moet aan het einde van een ronde k (na dat alle i zijn gedaan), met x_k[l][j] en dan klopt het vgm
                    
                else:
                    model.addConstr(gpy.quicksum(w_root[f,j] * x_prev[i][f] for f in range(nx))
                    + gpy.quicksum(w[f,j] * gpy.quicksum(b_prev[i][f][l] for l in sum_list) for f in range(nx)), 
                    gpy.GRB.EQUAL, output_vars[i][j] - s_i[j])

                # Constraints for ReLU logic
                if k != (num_layers - 1):                   
                    if hidden_bounds[k][j]:
                        # If we have bounds, we use the big-M formulation
                        lb, ub = hidden_bounds[k][j]
                        if ub >= 0:
                            model.addConstr(x_i[j] <= ub*z_i[j])
                        if lb <= 0:
                            model.addConstr(s_i[j] <= -lb*(1 - z_i[j]))
                    else:
                        # Otherwise, we use indicator constraints
                        model.addGenConstrIndicator(z_i[j], False, x_i[j] <= 0)
                        model.addGenConstrIndicator(z_i[j], True, s_i[j] <= 0)
                else:
                    if hidden_bounds[k][j]:
                        # If we have bounds, we use the big-M formulation
                        lb, ub = hidden_bounds[k][j]
                        if ub >= 0:
                            model.addConstr(output_vars[i][j] <= ub*z_i[j])
                        if lb <= 0:
                            model.addConstr(s_i[j] <= -lb*(1 - z_i[j]))
                    else:
                        # Otherwise, we use indicator constraints
                        model.addGenConstrIndicator(z_i[j], False, output_vars[i][j] <= 0)
                        model.addGenConstrIndicator(z_i[j], True, s_i[j] <= 0)
                    
            
            x_k.append(x_i)
            s_k.append(s_i)
            z_k.append(z_i)
            b_k.append(b_i)

        #op deze hoogte moeten helper constraints gemaakt worden.

        if k < num_layers:
            for i in range(n):
                for j in range(nu):
                    model.addConstrs((x_k[l][j] - hidden_bounds[k][j][1]*(1 - A[i,l]) <= b_k[i][j][l] for l in range(n)), name = f'24d_layer_{k}')
                    model.addConstrs((x_k[l][j] + hidden_bounds[k][j][1]*(1 - A[i,l]) >= b_k[i][j][l] for l in range(n)), name = f'24e_layer_{k}')
                    model.addConstrs((-hidden_bounds[k][j][1]*(A[i,l]) <= b_k[i][j][l] for l in range(n)), name = f'24f-a_layer_{k}')
                    model.addConstrs((hidden_bounds[k][j][1]*(A[i,l]) >= b_k[i][j][l] for l in range(n)), name = f'24f-b_layer_{k}')

        x_prev = x_k
        b_prev = b_k

    model.update()

    return model, output_vars




def build_graphconv_milp_and_run_bt(model: gpy.Model, input_vars, output_vars, state_dict, n, A, d_max, c_id=0, bt_procedures=None, bilinear = False, norm_term = None, multilayer = False):
    """
    Build a bound tightened MILP for a neural network

    Assuming that NN model has K hidden layers with ReLU activation and 1 linear output layer

    Variables are named as x_i_j_k, where i is the layer number, and j is the unit number (of layer i), and k is the
    constraint ID (c_id).

    TODO: add list of FBBT procedures to run (will be run in the order of the list)

    :param model: MILP model to add variables and constraints (Gurobi model)
    :param input_vars: List of input variables (Gurobi variables)
    :param output_vars: Output variables (Gurobi variables)
    :param nn_model: NN model (Keras model)
    :param c_id: Constraint ID (required to uniquely name variables when adding multiple NN models to MILP model)
    :param bt_procedures: Bound tightening procedures to run
    :return: model
    """
    # Get NN weights
    weights = get_weights(state_dict)

    F = int(len(input_vars) / n)
    input_vars = [[input_vars[i,j] for j in range(F)] for i in range(n)]
    input_bounds = [(v.getAttr(gpy.GRB.Attr.LB), v.getAttr(gpy.GRB.Attr.UB)) for v in input_vars[0]]

    # Tighten bounds
        
    output_bounds = [(v.getAttr(gpy.GRB.Attr.LB), 
                      v.getAttr(gpy.GRB.Attr.UB)) for v in output_vars[0]]
    bounds = run_bound_tightening(weights, input_bounds, output_bounds, bt_procedures, d_max)

    # Build NN constraints
    hidden_bounds = bounds[1:]

    model, output_vars = graph_conv_milp_builder(model, input_vars, output_vars, weights, n, A, hidden_bounds, c_id = 'gc')

    return model, output_vars, bounds


