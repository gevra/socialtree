import logging
import numpy as np
import networkx as nx
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly
import itertools
import time
import multiprocessing as mp
from common.functions import list2chunks, match_any_in_list2df_col

_logger = logging.getLogger(__name__)


def _betmap(G_normalized_weight_sources_tuple):
    """Pool for multiprocess only accepts functions with one argument.
    This function uses a tuple as its only argument. We use a named tuple for
    python 3 compatibility, and then unpack it when we send it to
    `betweenness_centrality_source`
    """
    return nx.betweenness_centrality_source(*G_normalized_weight_sources_tuple)


def betweenness_centrality_parallel(G, processes=None):
    """Parallel betweenness centrality  function"""
    p = mp.Pool(processes=processes)
    node_divisor = len(p._pool)*4
    node_chunks = list(list2chunks(G.nodes(), n=int(G.order()/node_divisor)))
    num_chunks = len(node_chunks)
    bt_sc = p.map(_betmap,
                  zip([G]*num_chunks,
                      [True]*num_chunks,
                      [None]*num_chunks,
                      node_chunks))

    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c


def patterns2graph(patterns_df, pop_thres=1):
    """
    Note that the popularity threshold is set to be the same for all length of patterns, which is not wise!!!
    :param patterns_df:
    :param pop_thres:
    :return:
    """
    g = nx.Graph()
    nodes_with_edges = set(
        [tag for p in
         patterns_df.loc[
             (patterns_df['support'] >= pop_thres) & (patterns_df['size'] == 2), 'pattern_list'
         ].tolist()
         for tag in p]
    )
    nodes_without_edges = [n for n in patterns_df.loc[
             (patterns_df['support'] >= pop_thres) & (patterns_df['size'] == 1), 'pattern'
         ].tolist() if n not in nodes_with_edges]
    graph_edges_list = patterns_df.loc[
        (patterns_df['support'] >= pop_thres) & (patterns_df['size'] == 2)
        ][['pattern_list', 'support']].apply(
        lambda p: (p['pattern_list'][0], p['pattern_list'][1], p['support']),
        axis=1
    )
    _logger.info(
        "there are %d nodes with %d edges + %d isolates with support>=%d" %
        (len(nodes_with_edges), len(graph_edges_list), len(nodes_without_edges), pop_thres)
    )
    g.add_weighted_edges_from(graph_edges_list)
    g.add_nodes_from(nodes_without_edges)
    isolates = list(nx.isolates(g))
    # sanity check
    # print("there are %d nodes with %d edges with support>=%d" % (len(g.nodes()), len(g.edges()), pop_thres))
    return g, nodes_with_edges, isolates


def get_node_features(G, list_of_node_features):
    graph_features_dict = {}
    if 'bet_cent' in list_of_node_features:
        graph_features_dict['bet_cent'] = nx.betweenness_centrality(G)  # change 'weight' to hashtag's inverse support, because the edge weights are currently treated as penalties/costs
    if 'deg_cent' in list_of_node_features:
        graph_features_dict['deg_cent'] = nx.degree_centrality(G)
    if 'eig_cent' in list_of_node_features:
        graph_features_dict['eig_cent'] = nx.eigenvector_centrality_numpy(G)
    if 'katz_cent' in list_of_node_features:
        graph_features_dict['katz_cent'] = nx.katz_centrality_numpy(G)
    if 'load_cent' in list_of_node_features:
        graph_features_dict['load_cent'] = nx.load_centrality(G)
    if 'pr' in list_of_node_features:
        graph_features_dict['pr'] = nx.pagerank(G)
    if 'degree' in list_of_node_features:
        graph_features_dict['degree'] = dict(nx.degree(G))
    return graph_features_dict


def get_node_features_mp(G, list_of_node_features, n_proc=None):
    if n_proc is None:
        n_proc = mp.cpu_count() - 1
    with mp.Pool(processes=min(n_proc, len(list_of_node_features))) as p:
        # if 'bet_cent' in list_of_node_features:
        bet_cent = betweenness_centrality_parallel(G, n_proc)
        _logger.debug("started a process to compute the betweenness centrality")
        # if 'deg_cent' in list_of_node_features:
        deg_cent = p.apply_async(nx.degree_centrality, args=(G,))
        _logger.debug("started a process to compute the degree centrality")
        # if 'eig_cent' in list_of_node_features:
        eig_cent = p.apply_async(nx.eigenvector_centrality_numpy, args=(G,))
        _logger.debug("started a process to compute the eigenvector centrality")
        # if 'katz_cent' in list_of_node_features:
        try:
            katz_cent = p.apply_async(nx.katz_centrality_numpy, args=(G,))
            _logger.debug("started a process to compute the katz centrality")
        except:
            list_of_node_features.remove('katz_cent')
        # if 'pr' in list_of_node_features:
        pr = p.apply_async(nx.pagerank, args=(G,))
        _logger.debug("started a process to compute the pagerank")
        # if 'degree' in list_of_node_features:
        degree = p.apply_async(nx.degree, args=(G,))
        print("started a process to compute the degree")
        # graph_features_list = [(f, locals()[f].get()) for f in list_of_node_features if f != 'bet_cent']
        # if 'bet_cent' in list_of_node_features:
        #     graph_features_list.append(('bet_cent', bet_cent))
        graph_features_dict = {
            'bet_cent': bet_cent, 'deg_cent': deg_cent.get(),
            'eig_cent': eig_cent.get(),
            # 'katz_cent': katz_cent.get(),
            # 'load_cent': load_cent.get(),
            'pr': pr.get(),
            'degree': dict(degree.get())
        }
    return graph_features_dict


def find_root(g, attr='weight', is_weight_positive=True, weighting_attr=None):
    if len(g.nodes()) == 1:
        return list(g.nodes())[0]
    else:
        if weighting_attr is not None:
            # NOTE that the 'weighting_attr' may be computed on the full graph, not the MST
            # e.g. the 'bet_cent' makes more sense to be computed not on the trimmed tree, but the graph
            node_importance_tuples = [
                (name, cumm_importance * g.node[name][weighting_attr]) for
                # (name, 0.5 / (1 / cumm_importance + np.sign(cumm_importance) / g.node[name][weighting_attr])) for
                (name, cumm_importance) in g.degree(weight=attr)
            ]
            # we also need to make sure that both have the same sign if added/subtracted e.g. in harmonic mean
            # the harmonic mean is a bad idea because it will ignore the importances
            # considering that they are orders of magnitude larger in scale than the centralities
        else:
            node_importance_tuples = list(g.degree(weight=attr))
        node_importance_tuples_sorted = sorted(node_importance_tuples, key=lambda x: x[1], reverse=is_weight_positive)
    _logger.debug(
        "the nodes with the highest (not mutually exclusive) cumulative importance overlaps %sare %s" % (
            "weighted by '%s' " % weighting_attr if weighting_attr is not None else "",
            str(node_importance_tuples_sorted[:5])
        )
    )
    return node_importance_tuples_sorted[0][0]


def set_tree_node_levels(t, root_nodes):
    current_level = 0
    nx.set_node_attributes(
        t,
        name='level',
        values=dict([(root_node, current_level) for root_node in root_nodes])
    )
    current_level += 1
    current_nodes = root_nodes
    parent_nodes = []
    while True:
        t, new_level, new_nodes = set_child_levels(t, current_level, current_nodes, parent_nodes)
        current_level = new_level
        if len(new_nodes) == 0:
            break
        parent_nodes = current_nodes
        current_nodes = new_nodes
    return t


def set_child_levels(t, current_level, current_nodes, parent_nodes):
    new_child_nodes = []
    for current_node in current_nodes:
        child_nodes = [node for node in t.neighbors(current_node) if node not in parent_nodes]
        nx.set_node_attributes(
            t,
            name='level',
            values=dict([(node, current_level) for node in child_nodes])
        )
        new_child_nodes.extend(child_nodes)
    return t, current_level+1, new_child_nodes


def set_tree_layout(t, prog=None):
    if prog == 'dot':
        root_node = sorted(t.nodes(data=True), key=lambda x: x[1]['level'], reverse=True)[0][0]
        nx.set_node_attributes(
            t,
            values=nx.drawing.nx_agraph.graphviz_layout(t, prog='dot', root=root_node),
            name='pos'
        )
    elif prog == 'spring':
        nx.set_node_attributes(
            t,
            name='pos',
            values=nx.spring_layout(
                t,
                iterations=100,  # default: 50
                # k=0.3,  # Optimal distance between nodes. If None the distance is set to 1/sqrt(n)
            )
        )
    else:
        t = set_simple_tree_layout(t)
    # print("--- the tree node positions are %s ---" % nx.get_node_attributes(t, 'pos'))
    return t


def set_simple_tree_layout(t, layer_distance=120):
    # the levels are set already, set the x axis coordinates
    # the graph must have the 'ts_start' & 'ts_end' attributes set
    forest_pos_values_list = []
    tree_y_offset = 0
    node_y_offset = -layer_distance / 6

    # determine the minimum distance (in seconds) between two nodes
    def determine_node_min_x_dist(g, plot_width_in_pixels=1760):
        all_ts_start_epoch = nx.get_node_attributes(g, 'ts_start_epoch').values()
        all_ts_end_epoch = nx.get_node_attributes(g, 'ts_end_epoch').values()
        node_min_x_dist_in_epochs = (max(all_ts_end_epoch) - min(all_ts_start_epoch)) / plot_width_in_pixels * 160
        # the choice of 60 is justified by the assumed size of node names on graph in order to avoid overlaps
        return node_min_x_dist_in_epochs
    node_min_x_dist = determine_node_min_x_dist(t)

    for tree in nx.connected_component_subgraphs(t):
        tree_pos_values_list = []
        for node_name, node_data in tree.nodes(data=True):
            # x = (node_data['ts_start_epoch'] + node_data['ts_end_epoch']) / 2
            x = np.mean(node_data['article_epochs'])
            y = layer_distance * node_data['level'] * (-1)**int(tree_y_offset == 0) + tree_y_offset
            # use the following line for all the trees to be on the same side
            # y = -layer_distance * node_data['level'] - tree_y_offset * int(tree_y_offset != 0)

            # check that there are no other nodes in close proximity
            while len([nn for (nn, (xx, yy)) in tree_pos_values_list + forest_pos_values_list if
                       abs(xx - x) < abs(node_min_x_dist) and abs(yy - y) < abs(node_y_offset)]) > 0:
                # make an "evasive" shift
                y += node_y_offset
                node_y_offset = layer_distance / 6 * (-1)**int(node_y_offset == 0)

            tree_pos_values_list.append((node_name, (x, y)))
        tree_y_offset = layer_distance / 12 * int(tree_y_offset == 0)
        forest_pos_values_list.extend(tree_pos_values_list)
    nx.set_node_attributes(
        t,
        values=dict(forest_pos_values_list),
        name='pos'
    )
    """
    the current implementation has a problem of substories overlapping completely if they appear at the same time
    consider a little y shift and some x shift as well
    """
    return t


def extract_tree_from_a_graph(g, patterns_df, incorporate_ts_length_days=False, node_score_dict=None, layout_prog=None,
                              importance_weighting_mode="prod/diff", story_distance_resolution=24*3600,
                              remove_isolates_flag=False):
    for u, v, d in g.edges(data=True):
        d['init_weight'] = d['weight']
        d['-weight'] = -d['weight']
        d['tree_weight'] = -d['weight']

        importance_u = 1
        importance_v = 1
        if incorporate_ts_length_days:
            importance_u *= g.node[u]['ts_n_dates']
            importance_v *= g.node[v]['ts_n_dates']
        if node_score_dict is not None:
            importance_u *= node_score_dict[u]
            importance_v *= node_score_dict[v]

        if importance_weighting_mode == 'sum':
            d['tree_weight'] *= importance_u + importance_v
        elif importance_weighting_mode == 'min':
            d['tree_weight'] *= min(importance_u, importance_v)
        elif importance_weighting_mode == 'hmean':
            d['tree_weight'] *= 1/2 * importance_u * importance_v / (importance_u + importance_v)
        elif importance_weighting_mode == 'prod/diff':
            d['tree_weight'] *= 1/2 * importance_u * importance_v / max(
                abs(importance_u - importance_v),
                0.5 * min(importance_u, importance_v)  # 0.5 is heuristically chosen and regulates the magnitude
            )  # just to make sure that the denominator is not 0 and is not sensitive to small importance differences
        else:
            raise Exception(
                "wrong importance weighting mode (%s) given... must be in ['sum', 'min', 'hmean', 'prod/diff']" %
                importance_weighting_mode
            )

        if story_distance_resolution:
            # story_center_of_mass_dist = abs(
            #     np.mean(g.node[u]['article_epochs']) - np.mean(g.node[v]['article_epochs'])
            # )
            story_median_dist = abs(np.median(g.node[u]['article_epochs']) - np.median(g.node[v]['article_epochs']))
            # the distances of stories' median articles is a more robust measure of story relatedness
            # than the distance of stories' centres of masses (article average timestamp)
            d['tree_weight'] *= story_distance_resolution / max(story_median_dist, story_distance_resolution)
            # the 'story_distance_resolution' is a normalization factor and max() makes the distances less than
            # 'story_distance_resolution' (24*3600 is a day) identical
            # connection edge weights of two stories with distances <='story_distance_resolution' are not discounted

            # our experiments show that the distances of stories' centres of mass or the median article distances
            # are better indicators of their relatedness than the overlap of their coverage periods

            # period_overlap = (
            #                          min(g.node[u]['ts_end_epoch'], g.node[v]['ts_end_epoch']) -
            #                          max(g.node[u]['ts_start_epoch'], g.node[v]['ts_start_epoch'])
            #                  ) / (
            #                          max(g.node[u]['ts_end_epoch'], g.node[v]['ts_end_epoch']) -
            #                          min(g.node[u]['ts_start_epoch'], g.node[v]['ts_start_epoch'])
            #                  )
            # d['tree_weight'] *= period_overlap

        # normalize and pass through an "activation function" to amplify or smoothen the differences
        # actually, no monotonic function will change the MST! https://stackoverflow.com/a/6690321/2262424

        # the assigning 'tree_weight' back to 'weight' is done only for the purpose of making many other functions
        # to operate without the need to know whether there is any other node attribute apart from 'weight'
        d['weight'] = -d['tree_weight']

    # remove nodes with no edges
    if remove_isolates_flag:
        # must be a list, not a generator, otherwise "RuntimeError: dictionary changed size during iteration"
        g.remove_nodes_from(list(nx.isolates(g)))

    # extract the MST
    t = nx.minimum_spanning_tree(g, weight='tree_weight')

    # set graph attributes
    list_of_graph_attributes = [
        'bet_cent', 'support', 'ts_length_days', 'ts_start', 'ts_end', 'ts_n_dates', 'article_epochs', 'score'
    ]
    # 'pr', 'pr_uw', 'pr_p', 'pr_p_uw', 'hub_degree', 'hub_degree_ratio',
    t = set_node_attr(t, patterns_df, list_of_graph_attributes)
    nx.set_edge_attributes(t, name='init_weight', values=nx.get_edge_attributes(g, 'init_weight'))
    nx.set_edge_attributes(t, name='redundant_nodes', values=nx.get_edge_attributes(g, 'redundant_nodes'))

    root_nodes = []
    for tree in nx.connected_component_subgraphs(t):
        root_nodes.append(find_root(tree, attr='tree_weight', is_weight_positive=False, weighting_attr='bet_cent'))

    # set the levels of the nodes given the roots and the edges in the MST
    t = set_tree_node_levels(t, root_nodes)
    t = set_tree_layout(t, prog=layout_prog)
    return t


def set_node_attr(g, patterns_df, list_of_graph_attributes):
    for attr in list_of_graph_attributes:
        nx.set_node_attributes(
            g,
            values=dict([
                (p['pattern'], p[attr]) for (i, p) in
                patterns_df[(patterns_df['size'] == 1) & (patterns_df['pattern'].isin(g.nodes()))].iterrows()
            ]),
            name=attr
        )
    return g


def eval_forest(t, patterns_df):
    """
    evaluate the tree branch disjointness and branch purity for each connected component separately
    :param t: undirected weighted graph (networkx.classes.graph.Graph instance) with nodes having "level" attributes
    :param patterns_df: a pandas dataframe containing the frequent tag co-occurrence patterns (as a ground truth)
    :return: list of the metrics from each connected component : weighted sum of cumulative disjointness and purity on
    all the graph levels separately for each connected component
    """
    forest_mi = {}
    for tree in nx.connected_component_subgraphs(t):
        root_node, mi = eval_tree(tree, patterns_df)
        forest_mi[root_node] = mi
    return forest_mi


def eval_tree(t, patterns_df):
    nodes_from_top_to_down = sorted(t.nodes(data=True), key=lambda x: x[1]['level'], reverse=False)
    root_node = nodes_from_top_to_down[0][0]
    deepest_level = nodes_from_top_to_down[-1][1]['level']
    mi_per_level = dict()
    for level in range(deepest_level, -1, -1):
        # compare the branches from the current level and below
        level_branches = get_comparison_branches(t, level)
        mi_per_level[level] = get_level_mi(level_branches, patterns_df, average=False)
    mi_cumulative = sum(mi for level, mi in mi_per_level.items())
    _logger.info("for the tree with root='%s' mi=%.2f" % (root_node, mi_cumulative))
    return root_node, mi_cumulative


def get_comparison_branches(t, level):
    # first get the nodes on the level, then expand by their connections
    branches = [[n] for n, d in t.nodes(data=True) if d['level'] == level]
    if len(branches) > 1:
        # getting all the children of the top nodes, in the beginning 'branch' contains a single node
        for i, branch in enumerate(branches):
            all_deep_children = get_tree_node_deep_children(t, branch[0])
            branches[i].extend(all_deep_children)
    # branches is a list of lists, because a single branch is a list of nodes
    return branches


def get_tree_node_deep_children(t, node):
    current_nodes = [node]
    all_deep_children = []
    while True:
        new_children_all_branches = get_immediate_children_in_tree(t, current_nodes)
        if len(new_children_all_branches) == 0:
            break
        else:
            all_deep_children.extend(new_children_all_branches)
            current_nodes = new_children_all_branches
    return all_deep_children


def get_immediate_children_in_tree(t, nodes):
    child_nodes = []
    for n in nodes:
        child_nodes.extend([child for child in t.neighbors(n) if t.node[child]['level'] > t.node[n]['level']])
    return child_nodes


def get_level_mi(level_branches, patterns_df, average=False):
    if len(level_branches) == 1:
        return 0
    patterns_df_reindexed = patterns_df[patterns_df['size'] == 1].set_index('pattern')
    mi_list = [get_branch_pair_pmi(pair, patterns_df_reindexed) for pair in itertools.combinations(level_branches, 2)]
    mi = sum(mi_list)
    if average:
        mi = mi / len(mi_list)
    return mi


def get_branch_pair_pmi(branch_pair, patterns_df_reindexed):
    """

    :param branch_pair: a pair of branches, each being a list of hashtags
    :param patterns_df_reindexed: the dataframe holding the extracted pattern information with 'pattern' as the index
    :return: mutual information of the branch pair
    """
    docs_branch_0 = set([
        a_id for node_recall in [patterns_df_reindexed.loc[node]['article_ids'] for node in branch_pair[0]]
        for a_id in node_recall
    ])
    docs_branch_1 = set([
        a_id for node_recall in [patterns_df_reindexed.loc[node]['article_ids'] for node in branch_pair[1]]
        for a_id in node_recall
    ])
    pmi = len(set(docs_branch_0) & set(docs_branch_1)) / len(set(docs_branch_0)) / len(set(docs_branch_1))
    return pmi


def create_edge_trace_for_plotly(g, middle_node_trace, min_edge_weight, max_edge_weight, color, dash=None,
                                 smart_layout=False):
    edge_traces = []
    for n1, n2, edge_data in g.edges(data=True):
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=go.scatter.Line(
                width=max(1, 5 * (edge_data['weight'] - min_edge_weight) / (max_edge_weight - min_edge_weight)),
                color=color, dash=dash
            ),
            hoverinfo='none',
            mode='lines'
        )
        x0, y0 = g.node[n1]['pos']
        x1, y1 = g.node[n2]['pos']
        if smart_layout:
            # x0 = (g.node[n1]['ts_start_epoch'] + g.node[n1]['ts_end_epoch']) / 2 * 1000
            # x1 = (g.node[n2]['ts_start_epoch'] + g.node[n2]['ts_end_epoch']) / 2 * 1000
            x0 = np.mean(g.node[n1]['article_epochs']) * 1000
            x1 = np.mean(g.node[n2]['article_epochs']) * 1000
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
        edge_traces.append(edge_trace)

        if edge_data['weight'] > 0:
            # middle_node_trace['x'].append((x0 + x1) / 2)
            # middle_node_trace['y'].append((y0 + y1) / 2)
            middle_node_trace['x'] += ((x0 + x1) / 2, )
            middle_node_trace['y'] += ((y0 + y1) / 2, )
            middle_node_trace['text'] += (" ".join([
                "%s-%s" % (n1, n2),
                "%.2f" % edge_data['weight'] if isinstance(edge_data['weight'], float) else str(
                    edge_data['weight']),
                "<br>%d articles" % edge_data['init_weight'] if 'init_weight' in edge_data else ""
            ]), )
    return edge_traces, middle_node_trace


def plot_a_graph(g, secondary_edges=None, title=None, smart_sizing=True, smart_shaping=True, smart_layout=False,
                 mark_root=False, interactive_mode_flag=True, save_image_path=None):
    if title is None:
        title = 'StoryGraph<br>'
    node_color = 'rgba(152, 76, 0, .1)'
    root_color = 'rgba(0, 152, 0, .15)'
    if 'pos' not in list(g.nodes(data=True))[0][1].keys():
        nx.set_node_attributes(
            g,
            name='pos',
            values=nx.spring_layout(
                g,
                iterations=20,  # default: 50
                k=0.9,  # Optimal distance between nodes. If None the distance is set to 1/sqrt(n)
            )
        )
    edge_traces = []
    node_violin_traces = []
    middle_node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=go.scatter.Marker(
            opacity=0,
        ),
        textposition='top right'
    )
    edge_weights = [e[2]['weight'] for e in g.edges(data=True) if e[2]['weight'] > 0]
    min_edge_weight = min(edge_weights)
    max_edge_weight = max(edge_weights)
    if max_edge_weight == min_edge_weight:  # corner-case
        min_edge_weight = 0
    primary_edge_traces, middle_node_trace = create_edge_trace_for_plotly(
        g, middle_node_trace=middle_node_trace, min_edge_weight=min_edge_weight,
        max_edge_weight=max_edge_weight, color='#888', dash=None, smart_layout=smart_layout
    )
    edge_traces.extend(primary_edge_traces)

    if secondary_edges is not None:
        secondary_edge_traces, middle_node_trace = create_edge_trace_for_plotly(
            secondary_edges, middle_node_trace=middle_node_trace, min_edge_weight=min_edge_weight,
            max_edge_weight=max_edge_weight, color='#0ff', dash='dash', smart_layout=smart_layout
        )
        edge_traces.extend(secondary_edge_traces)

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        hovertext=[],
        mode='markers+text',
        textposition='middle center',
        textfont={"size": []},
        hoverinfo='text',
        marker=go.scatter.Marker(
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=[],
        )
    )

    layout = go.Layout(
        autosize=False,
        height=(720 if interactive_mode_flag else 960) * np.ceil(len(list(g.nodes())) / 100),
        width=1320 if interactive_mode_flag else 1760,
        title=title,
        titlefont=dict(size=16),
        font=dict(size=18),  # 24 for saving the fig
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=10, r=10, t=60 + 20 * title.count("<br>")),
        xaxis=go.layout.XAxis(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=go.layout.YAxis(showgrid=False, zeroline=False, showticklabels=False),
        shapes=[]
    )
    if smart_layout:
        layout['xaxis'] = go.layout.XAxis(type="date", showgrid=True, ticklen=5)
        layout['margin'] = dict(b=60, l=5, r=5, t=60 + 20 * title.count("<br>"))

    # first_trace_flag = True  # modified this after switching to plotly 3.3
    for node_name, node_data in g.nodes(data=True):
        x, y = node_data['pos']
        # degree = g.degree(node, weight=None)
        if smart_layout:
            # x = (node_data['ts_start_epoch'] + node_data['ts_end_epoch']) / 2 * 1000
            x = np.mean(node_data['article_epochs']) * 1000
        if smart_sizing:
            if (not smart_shaping) or (len(set(node_data['article_epochs'])) < 3):
                layout['shapes'] += ({
                    'type': 'circle',
                    'xref': 'x',
                    'yref': 'y',
                    'x0': node_data['ts_start_epoch'] * 1000,
                    'y0': y - max(10, 4 * np.sqrt(node_data['support'])) / 2,
                    'x1': node_data['ts_end_epoch'] * 1000,
                    'y1': y + max(10, 4 * np.sqrt(node_data['support'])) / 2,
                    'fillcolor': root_color if mark_root and node_data['level'] == 0 else node_color,
                    'line': {
                        'width': 0.1,
                    },
                }, )
            else:
                # https://github.com/plotly/plotly.py/blob/master/plotly/figure_factory/_violin.py
                if len(node_data['article_epochs']) > 1:
                    node_violin_trace = ff.create_violin(
                        [
                            (article_epoch - node_data['ts_start_epoch']) /
                            (node_data['ts_end_epoch'] - node_data['ts_start_epoch'])
                            for article_epoch in node_data['article_epochs']
                        ], rugplot=False
                    )['data']
                    node_violin_trace = node_violin_trace[:2]  # the first two Scatter objects are the top/bottom halves
                    first_trace_flag = True  # dirty trick! modified this after switching to plotly 3.3
                    for violin_trace in node_violin_trace:
                        temp = np.copy(violin_trace['x'])
                        violin_trace['x'] = 1000 * (
                            (node_data['ts_end_epoch'] - node_data['ts_start_epoch']) * np.copy(violin_trace['y']) +
                            node_data['ts_start_epoch']
                        )
                        violin_trace['y'] = y + 2 * np.sqrt(node_data['support']) * temp
                        if 'box' in violin_trace:
                            violin_trace['box'] = {'visible': False}
                        if 'scalemode' in violin_trace:
                            violin_trace['scalemode'] = 'count'
                        if 'meanline' in violin_trace:
                            violin_trace['meanline'] = {'visible': False}
                        if 'opacity' in violin_trace:
                            violin_trace['opacity'] = 0.05
                        if 'fill' in violin_trace:
                            if first_trace_flag:
                                violin_trace['fill'] = 'toself'
                                first_trace_flag = False
                            else:
                                violin_trace['fill'] = 'tonexty'
                        if 'text' in violin_trace:
                            violin_trace['text'] = None
                        if 'line' in violin_trace:
                            violin_trace['line']['color'] = 'rgba(0,0,0,0)'
                        if 'marker' in violin_trace:
                            violin_trace['marker']['color'] = 'rgba(0,0,0,0)'
                        if 'fillcolor' in violin_trace:
                            violin_trace['fillcolor'] = root_color if mark_root and node_data['level'] == 0 else node_color
                    node_violin_traces.extend(node_violin_trace)
        else:
            if mark_root and node_data['level'] == 0:
                node_trace['marker']['color'] += (root_color, )
            else:
                node_trace['marker']['color'] += (node_color, )  # node_data['bet_cent']
            if smart_shaping:
                node_trace['marker']['size'] += (max(50, 20 * np.sqrt(node_data['support'])), )  # g.node[node]['pr']
            else:
                node_trace['marker']['size'] += (50, )
        node_trace['x'] += (x, )
        node_trace['y'] += (y, )
        node_trace['hovertext'] += ("<br>".join([
            # "%s" % node_name,
            "%d article%s" % (node_data['support'], "s" if node_data['support'] > 1 else ""),
            "query relevance: %.4f" % node_data['score'],
            ("level: %d" % node_data['level']) if 'level' in node_data.keys() else "",
            "%d day%s" % (node_data['ts_n_dates'], "s" if node_data['ts_n_dates'] > 0 else ""),
            "%s - %s" % (
                time.strftime('%d.%m.%Y', time.localtime(node_data['ts_start_epoch'])),
                time.strftime('%d.%m.%Y', time.localtime(node_data['ts_end_epoch']))
            ),
            ("redundant tags: %s" % node_data['redundant_nodes']) if 'redundant_nodes' in node_data.keys() else ""
            # "%s<br>(%.1f %.1f)" % (node_name, x, y)
        ]), )
        node_trace['text'] += (
            "%s" % node_name,
            # "%s<br>(%.1f %.1f)" % (node_name, x, y)
        )  # + " (degree: %d) (hub_degree: %d)" % (T.degree(node), int(T.node[node]['hub_degree'])))
        node_trace['textfont']['size'] += (max(12, 4 * np.log2(node_data['support'])), )

    fig = go.Figure(
        data=[*node_violin_traces, *edge_traces, node_trace, middle_node_trace],
        layout=layout
    )
    if interactive_mode_flag:
        div = plotly.offline.iplot(
            fig,
            image_height=720 * np.ceil(len(list(g.nodes())) / 100),
            image_width=1320,
        )
    else:
        div = plotly.offline.plot(
            fig,
            auto_open=False,
            output_type='div',
            image_height=960 * np.ceil(len(list(g.nodes())) / 100),
            image_width=1760,
        )
    # save the plot(s)
    if save_image_path is not None:
        # remove the title
        fig['layout']['title'] = None
        fig['layout']['margin'] = dict(b=60, l=40, r=40, t=60)
        plotly.offline.plot(
            fig,
            filename=save_image_path,
            image='svg',
            auto_open=False,
            image_height=480,  # 960,
            image_width=1200,  # 1760
        )
    return div


def extract_and_plot_forest(g, patterns_df, articles_df, field_name, tag_profile, importance_weighting_mode,
                            incorporate_ts_length_days, story_distance_resolution, layout_prog, remove_isolates_flag,
                            smart_sizing=False, smart_shaping=True, smart_layout=False, timing_marks=None,
                            save_image_path=None):
    t_type = "_".join(["time-aware"] * incorporate_ts_length_days + ["query-aware"] * (tag_profile is not None))
    if timing_marks is not None:
        timing_marks.append(('extract_%s_tree_start' % t_type, time.time()))
    start_time = time.time()
    t = extract_tree_from_a_graph(
        g, patterns_df,
        incorporate_ts_length_days=incorporate_ts_length_days,
        node_score_dict=tag_profile, layout_prog=layout_prog, importance_weighting_mode=importance_weighting_mode,
        story_distance_resolution=story_distance_resolution, remove_isolates_flag=remove_isolates_flag
    )

    # find the removed edges
    # nx.difference(g, t)  # Attributes from the graph, nodes, and edges are not copied to the new graph.
    removed_edges = g.copy()
    removed_edges.remove_edges_from(e for e in g.edges() if e in t.edges())
    # print("the removed edges are %s" % list(removed_edges.edges()))
    # add the new node coordinates that the original graph didn't have, but the tree has (e.g. 'pos')
    nx.set_node_attributes(removed_edges, name='pos', values=nx.get_node_attributes(t, 'pos'))

    t_recall = articles_df[articles_df[[field_name]].apply(
        func=match_any_in_list2df_col,
        args=(t.nodes(), field_name,),
        axis=1
    )].shape[0]
    end_time = time.time()
    if timing_marks is not None:
        _logger.info("--- extracted the %s MST in %.3f seconds ---" % (t_type, time.time() - timing_marks[-1][1]))
        timing_marks.append(('extract_%s_tree' % t_type, time.time()))

    tree_title_form = (
        "<br>%s-based %s summary tree (%.4f secs)"
        # "<br>mi=%s"
        "<br>trees (roots): <b>%d</b> (%d of which are isolates)     nodes: <b>%d</b>     edges: %d     "
        "cumulative weight: <b>%.2f</b>     recall: <b>%.2f%%</b>"
        "<br>extracted from <b>%d</b> articles"
    )
    t_div = plot_a_graph(
        t,
        # secondary_edges=removed_edges,
        title=tree_title_form % (
            field_name, t_type,
            end_time - start_time,
            # dict([(root, "%.4f" % mi) for (root, mi) in eval_forest(t, patterns_df).items()]),
            nx.number_connected_components(t), len(list(nx.isolates(g))), len(t.nodes()),
            len([e for e in t.edges(data=True) if e[2]['tree_weight'] != 0]),
            sum(s for (t, s) in tag_profile.items()) if tag_profile is not None else 0,
            t_recall / articles_df.shape[0] * 100, articles_df.shape[0]
        ),
        smart_sizing=smart_sizing, smart_shaping=smart_shaping, smart_layout=smart_layout, mark_root=True,
        save_image_path=save_image_path
    )
    if timing_marks is not None:
        _logger.info(
            "--- plotted the %s MST in %.3f seconds (%.1f seconds since start) ---" %
            (t_type, time.time() - timing_marks[-1][1], time.time() - timing_marks[0][1])
        )
        timing_marks.append(('plot_%s_tree_%s' % (t_type, field_name), time.time()))
    return t_div
