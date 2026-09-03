import numpy as np
import pandas as pd
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint
from collections import deque


def get_node_by_name(graph: GeneralGraph, name: str, labels: list) -> GraphNode:
    """Finds the GraphNode object corresponding to a given name."""
    try:
        index = labels.index(name)
        node_internal_name = f"X{index + 1}"
        return graph.get_node(node_internal_name)
    except (ValueError, KeyError):
        print(f"Warning: Node '{name}' not found in graph labels.")
        return None

def get_name_by_node(node: GraphNode, labels: list) -> str:
    """Finds the name corresponding to a GraphNode object."""
    try:
        index = int(node.get_name().replace("X", "")) - 1
        return labels[index]
    except (ValueError, IndexError):
        print(f"Warning: Could not map node '{node.get_name()}' back to a label.")
        return None

def get_neighbors(graph: GeneralGraph, node: GraphNode) -> list[GraphNode]:
    """Gets all adjacent nodes."""
    return graph.get_adjacent_nodes(node)

def get_parents(graph: GeneralGraph, node: GraphNode) -> list[GraphNode]:
    """Gets definite and potential parents (nodes with edge ending in ARROW or CIRCLE at 'node')."""
    parents = []
    for edge in graph.get_graph_edges():
        if edge.get_node2() == node:
            if edge.get_endpoint1() in [Endpoint.ARROW, Endpoint.CIRCLE]:
                 if edge.get_endpoint2() == Endpoint.ARROW:
                     parents.append(edge.get_node1())
                 elif edge.get_endpoint2() == Endpoint.CIRCLE:
                     parents.append(edge.get_node1())

        elif edge.get_node1() == node:
            if edge.get_endpoint2() in [Endpoint.ARROW, Endpoint.CIRCLE]:
                 if edge.get_endpoint1() == Endpoint.ARROW:
                     parents.append(edge.get_node2())
                 elif edge.get_endpoint1() == Endpoint.CIRCLE:
                     parents.append(edge.get_node2())

    return list(set(parents))

def get_children(graph: GeneralGraph, node: GraphNode) -> list[GraphNode]:
    """Gets definite and potential children (nodes with edge starting with ARROW or CIRCLE from 'node')."""
    children = []
    for edge in graph.get_graph_edges():
        if edge.get_node1() == node:
            if edge.get_endpoint1() == Endpoint.ARROW:
                 if edge.get_endpoint2() == Endpoint.ARROW:
                     children.append(edge.get_node2())
                 elif edge.get_endpoint2() == Endpoint.TAIL:
                     children.append(edge.get_node2())
                     
            elif edge.get_endpoint1() == Endpoint.CIRCLE:
                if edge.get_endpoint2() in [Endpoint.ARROW, Endpoint.CIRCLE]:
                    children.append(edge.get_node2())

        elif edge.get_node2() == node:
            if edge.get_endpoint2() == Endpoint.ARROW:
                 if edge.get_endpoint1() == Endpoint.ARROW:
                     children.append(edge.get_node1())
            elif edge.get_endpoint2() == Endpoint.CIRCLE:
                 if edge.get_endpoint1() == Endpoint.CIRCLE:
                     children.append(edge.get_node1())
                 elif edge.get_endpoint1() == Endpoint.ARROW:
                     children.append(edge.get_node1())


    return list(set(children)) # Return unique nodes

def get_descendants(graph: GeneralGraph, start_node: GraphNode, labels: list) -> set[str]:
    """Finds all descendants reachable via directed paths (->, o->, o-o)."""
    descendants = set()
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()
        children = get_children(graph, current_node)

        for child in children:
            if child not in visited:
                child_name = get_name_by_node(child, labels)
                if child_name:
                    descendants.add(child_name)
                visited.add(child)
                queue.append(child)

    return descendants

def get_non_descendants(graph: GeneralGraph, start_node: GraphNode, labels: list) -> set[str]:
    """
    Finds all nodes that are NOT descendants of start_node (and not start_node itself).
    Descendants are nodes reachable via potentially directed paths (->, o->, o-o).
    """
    if start_node is None:
        return set(labels) # Or handle error appropriately

    start_node_name = get_name_by_node(start_node, labels)
    if start_node_name is None:
        return set(labels) # Or handle error

    descendant_names = get_descendants(graph, start_node, labels)

    all_node_names = set(labels)
    non_descendant_names = all_node_names - descendant_names - {start_node_name}

    return non_descendant_names


def validate_causal_consistency(
    cf_row: pd.Series,
    original_row: pd.Series,
    graph: GeneralGraph,
    labels: list,
    threshold_independent_change: float = 0.4,
    epsilon: float = 1e-6
) -> tuple[bool, dict]:
    """
    Validates a counterfactual sample based on structural causal consistency with the graph.
    Checks if the intervention occurred and if nodes expected to be independent remain unchanged.

    Args:
        cf_row: The counterfactual data row (must contain 'scenario' column with intervened factor name).
        original_row: The original data row corresponding to cf_row.
        graph: The estimated causal graph (GeneralGraph object).
        labels: List of factor names corresponding to graph nodes.
        threshold_independent_change: Max proportion of non-descendant nodes allowed to change.
        epsilon: Tolerance level to consider a variable as changed.

    Returns:
        tuple: (is_consistent: bool, details: dict)
    """
    details = {}
    intervened_factor_name = cf_row.get('scenario', None)

    if intervened_factor_name is None:
        details['error'] = "Counterfactual row missing 'scenario' column."
        return False, details

    intervened_node = get_node_by_name(graph, intervened_factor_name, labels)

    if intervened_node is None:
        details['warning'] = f"Intervened factor '{intervened_factor_name}' not found in graph. Skipping consistency check."
        details['intervention_success'] = 'N/A'
        details['independent_check'] = 'N/A'
        return True, details

    original_value = original_row.get(intervened_factor_name, None)
    predicted_value = cf_row.get(intervened_factor_name, None)

    if original_value is None or predicted_value is None:
        details['error'] = f"Missing value for intervened factor '{intervened_factor_name}' in original or counterfactual row."
        return False, details

    try:
        original_value = float(original_value)
        predicted_value = float(predicted_value)
        delta_intervened = predicted_value - original_value
    except (ValueError, TypeError):
        details['error'] = f"Cannot convert values for '{intervened_factor_name}' to numeric."
        return False, details

    if abs(delta_intervened) < epsilon:
         details['intervention_success'] = False
         details['reason'] = f"Intervened factor '{intervened_factor_name}' did not change value significantly ({original_value} -> {predicted_value})."
         return False, details
    else:
         details['intervention_success'] = True

    non_descendant_names = get_non_descendants(graph, intervened_node, labels)
    details['non_descendants_checked'] = list(non_descendant_names)

    changed_independent_count = 0
    checked_independent_count = 0
    changed_independent_nodes_details = []

    if non_descendant_names:
        for name in non_descendant_names:
            if name in ['score', 'diagnosis']:
                continue
            orig_y = original_row.get(name, None)
            pred_y = cf_row.get(name, None)

            if orig_y is None or pred_y is None:
                print(f"Warning: Missing value for non-descendant '{name}'. Skipping check for this node.")
                continue

            try:
                orig_y = float(orig_y)
                pred_y = float(pred_y)
            except (ValueError, TypeError):
                print(f"Warning: Cannot convert values for non-descendant '{name}' to numeric. Skipping check.")
                continue

            checked_independent_count += 1
            delta_y = pred_y - orig_y

            if abs(delta_y) > epsilon:
                changed_independent_count += 1
                changed_independent_nodes_details.append({
                    "name": name,
                    "original": orig_y,
                    "predicted": pred_y,
                    "delta": delta_y
                })

        independent_change_ratio = (changed_independent_count / checked_independent_count) if checked_independent_count > 0 else 0
        details['independent_check'] = {
            "checked_count": checked_independent_count,
            "changed_count": changed_independent_count,
            "change_ratio": independent_change_ratio,
            "changed_nodes": changed_independent_nodes_details
        }

        if independent_change_ratio > threshold_independent_change:
            details['reason'] = f"High change in independent nodes ({independent_change_ratio:.2f} > {threshold_independent_change})."
            return False, details
    else:
         details['independent_check'] = {"checked_count": 0, "changed_count": 0, "change_ratio": 0.0}

    details['reason'] = "Passed causal consistency checks (intervention success, low change in independent nodes)."
    return True, details