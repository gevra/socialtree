import logging
import time
import numpy as np
import networkx as nx
from pathlib import Path
from profile_functions import get_reduced_tag_profile_and_stats
from common.functions import match_any_in_list2df_col
from common.pattern_functions import export_docs_for_eclat, mine_patterns_with_eclat, \
    populate_pattern_support_doc_ids, plot_pattern_stats
from common.graph_functions import patterns2graph, get_node_features_mp, get_node_features, set_node_attr, \
    extract_and_plot_forest, plot_a_graph

_logger = logging.getLogger(__name__)


########################################################################################################################
# ------------------------------------------------- THE MAIN FUNCTIONS ----------------------------------------------- #
########################################################################################################################
def visualize_trees_and_graphs(story_id, articles_df, timing_marks, story_info,
                               tag_profile_field_name, tag_blacklist, redundancy_conf_thres, min_profile_size,
                               lexical_redundancy_prefix, lexical_redundancy_suffix,
                               plot_stats_flag, tag_list_field_name, tag_field_blacklist, visualize_tree_flag,
                               importance_weighting_mode, story_distance_resolution,
                               pattern_type, min_pattern_support, visualize_intermediate_trees_flag):
    _logger.info("-" * 26 + " EXTRACTING THE TAG PROFILE " + "-" * 26)
    tag_score_dict, redundant_tags_dict, tag_recall_div, tag_profile_div, timing_marks = \
        get_reduced_tag_profile_and_stats(
            articles_df=articles_df,
            tag_profile_field_name=tag_profile_field_name,
            tag_blacklist=tag_blacklist,
            story_id=story_id,
            timing_marks=timing_marks,
            min_article_recall=None,
            redundancy_conf_thres=redundancy_conf_thres,
            lexical_redundancy_prefix=lexical_redundancy_prefix,
            lexical_redundancy_suffix=lexical_redundancy_suffix,
            min_profile_size=min_profile_size,
            normalized=True,
            plot_stats_flag=plot_stats_flag
        )
    articles_df[tag_list_field_name + "_original"] = articles_df[tag_list_field_name]
    articles_df[tag_list_field_name] = articles_df[tag_profile_field_name].map(lambda p: list(p.keys()))
    story_info += "<p><h4>story's profile (%d tags, %.2f cumulative weight):</h4> " % \
                  (len(tag_score_dict.items()), sum(tag_score_dict.values())) + \
                  ", ".join(
                      t + ": %.2f" % s for t, s in
                      sorted(tag_score_dict.items(), key=lambda x: x[1], reverse=True)
                  ) + "</p>\n\n"
    _logger.info("-" * 26 + " EXTRACTED THE TAG PROFILE " + "-" * 27)

    _logger.info("-" * 30 + " CREATING A TAG TREE " + "-" * 29)
    patterns_df, tag_support_div, timing_marks, g_div, t_plain_div, t_days_div, t_tag_profile_div, \
        t_basic_div, t_simple_div, t_div = extract_and_plot_patterns(
            articles_df=articles_df.copy(),
            field_name=tag_list_field_name,
            field_blacklist=tag_field_blacklist,
            pattern_type=pattern_type,
            story_id=story_id,
            visualize_tree_flag=visualize_tree_flag,
            importance_weighting_mode=importance_weighting_mode,
            story_distance_resolution=story_distance_resolution,
            timing_marks=timing_marks,
            min_pattern_support=min_pattern_support,
            tag_profile=tag_score_dict,
            redundant_tags_dict=redundant_tags_dict,
            plot_pattern_stats_flag=plot_stats_flag,
            visualize_intermediate_trees_flag=visualize_intermediate_trees_flag
        )
    _logger.info("-" * 30 + " CREATED A TAG TREE " + "-" * 30)
    return tag_score_dict, timing_marks, story_info, \
           tag_recall_div, tag_profile_div, tag_support_div, g_div, t_plain_div, t_days_div, t_tag_profile_div, \
           t_basic_div, t_simple_div, t_div


def extract_and_plot_patterns(articles_df, field_name, field_blacklist, story_id, timing_marks,
                              pattern_type='frequent', min_pattern_support=-1,
                              importance_weighting_mode='sum', story_distance_resolution=24*3600,
                              tag_profile=None, redundant_tags_dict=None, visualize_tree_flag=False,
                              plot_pattern_stats_flag=False, visualize_intermediate_trees_flag=False):
    """

    :param plot_pattern_stats_flag:
    :param min_pattern_support:
    :param articles_df:
    :param field_name: takes values in ['tag_list', 'good_hashtags', 'keywords_list']
    :param story_id:
    :param field_blacklist:
    :param pattern_type: takes values in ['frequent', 'closed', 'maximal', 'generators, 'rules']
    :param visualize_tree_flag:
    :param importance_weighting_mode: default: 'prod/diff' # takes values in ['sum', 'min', 'hmean', 'prod/diff']
    :param story_distance_resolution: time resolution in seconds, doesn't incorporate substory distances if None or 0
    :param timing_marks:
    :param tag_profile:
    :param redundant_tags_dict: Note that this is used only in the visualization and does not affect support calculation
    :param visualize_intermediate_trees_flag
    :return:
    """

    # ------------------------------------------------------------------------------------------------------------------
    # because the indexing and support retrieval is done by matching a string, the multi-grams must be made unigrams
    articles_df[field_name + "_"] = articles_df[field_name].map(lambda kws: ["_".join(kw.split(" ")) for kw in kws])
    tag_profile = dict([("_".join(kw.split(" ")), s) for (kw, s) in tag_profile.items()])

    # extract the patterns
    infile_name = export_docs_for_eclat(
        list_of_docs=articles_df.to_dict('records'), item_field_name=field_name + "_",
        item_profile_list=[t for t, score in tag_profile.items()],
        collection_id=story_id, field_blacklist=field_blacklist
    )
    outfile_name = Path("patterns", "story_patterns_%s_%s_%s.out" % (story_id, field_name, pattern_type))
    patterns_df = mine_patterns_with_eclat(
        eclat_path=Path('eclat').resolve(),
        infile_name=infile_name, outfile_name=outfile_name, pattern_type=pattern_type, support=min_pattern_support
    )
    _logger.info(
        "--- mined frequent patterns in %.3f seconds (%.1f seconds since start) ---" %
        (time.time() - timing_marks[-1][1], time.time() - timing_marks[0][1])
    )
    timing_marks.append(('extract_patterns', time.time()))

    # ------------------------------------------------------------------------------------------------------------------
    # plot pattern stats
    if plot_pattern_stats_flag:
        tag_support_div = plot_pattern_stats(patterns_df=patterns_df, field_name=field_name)
    else:
        tag_support_div = ""

    # ------------------------------------------------------------------------------------------------------------------
    # visualize tree(s)
    g_div, t_plain_div, t_days_div, t_tag_profile_div, t_simple_div, t_basic_div, t_div = "", "", "", "", "", "", ""
    if visualize_tree_flag:
        _logger.info("--- starting the graph construction and the tree extraction ---")
        timing_marks.append(('construct_graph_start', time.time()))
        # get the support article ids for the patterns
        patterns_df = populate_pattern_support_doc_ids(
            patterns_df=patterns_df,
            docs_df=articles_df,
            field_name=field_name + "_"
        )
        patterns_df = enrich_pattern_features(patterns_df, articles_df[['id', 't', 'epoch']])

        # construct a graph
        timing_marks.append(('construct_graph_start', time.time()))
        g, nodes_with_edges, isolates = patterns2graph(patterns_df, pop_thres=1)
        _logger.info(
            "--- constructed the graph in %.3f seconds (%.1f seconds since start) ---" %
            (time.time() - timing_marks[-1][1], time.time() - timing_marks[0][1])
        )
        timing_marks.append(('construct_graph', time.time()))

        # get the graph features (node centralities, degrees)
        list_of_node_features = ['bet_cent', 'deg_cent', 'pr', 'degree']  # only 'bet_cent' is needed for SocialTree
        """
        getting the node features for a graph with 22766 nodes and with 113410 edges, 
        takes ~18 minutes with parallel execution on 10 cores (~2 hours 25 minutes on a single core) 
        getting the node features for a graph with 11157 nodes and with 82323 edges, takes ~40 minutes on a single core
        """
        if len(nodes_with_edges) > 100:
            _logger.debug("started parallel processes for graph node feature extraction")
            node_features_dict = get_node_features_mp(g, list_of_node_features)
        else:
            node_features_dict = get_node_features(g, list_of_node_features)
        for f in list_of_node_features:
            patterns_df[f] = np.nan
            # the if condition is needed because the praph may be filtered by pop_thres
            patterns_df.loc[patterns_df['size'] == 1, f] = patterns_df[patterns_df['size'] == 1]['pattern'].map(
                lambda p: node_features_dict[f][p] if (p in node_features_dict[f].keys()) else np.nan
            )
        if tag_profile is None:
            tag_profile = dict((n, 1) for n in g.nodes())
        patterns_df.loc[patterns_df['pattern'].isin(g.nodes()), 'score'] = \
            patterns_df[patterns_df['pattern'].isin(g.nodes())]['pattern'].map(lambda x: tag_profile[x])

        g = set_node_attr(
            g,
            patterns_df,
            ['bet_cent', 'pr', 'support', 'ts_length_days', 'ts_start', 'ts_end', 'ts_start_epoch', 'ts_end_epoch',
             'ts_n_dates', 'article_epochs', 'score']
        )
        # print("the redundant tag sets are: %s" % redundant_tags_dict)
        nx.set_node_attributes(g, name='redundant_nodes', values=redundant_tags_dict)
        _logger.info(
            "--- set the node attributes in %.3f seconds (%.1f seconds since start) ---" %
            (time.time() - timing_marks[-1][1], time.time() - timing_marks[0][1])
        )
        timing_marks.append(('set_node_attributes', time.time()))
        # calculate the graph nodes' recall
        g_recall = articles_df[articles_df[[field_name + "_"]].apply(
            func=match_any_in_list2df_col,
            list_of_tags=g.nodes(), field_name=field_name + "_",
            axis=1
        )].shape[0]
        g_div = plot_a_graph(
            g,
            title="SocialGraph (%.4f secs)"
                  "<br>nodes: %d (%d of which are isolates)     edges: %d     recall: %.2f%%"
                  "<br>extracted from %d articles" % (
                      dict(timing_marks)['set_node_attributes'] - dict(timing_marks)['construct_graph_start'],
                      len(g.nodes()), len(isolates),
                      len([e for e in g.edges(data=True) if e[2]['weight'] != 0]),
                      g_recall / articles_df.shape[0] * 100, articles_df.shape[0]
                  ),
            smart_sizing=False,
            save_image_path=str(
                Path("../interactive_summaries/%s_%s-graph.html" % (story_id, field_name)).resolve()
            )
        )

        # get a tree
        if visualize_intermediate_trees_flag:
            t_plain_div = extract_and_plot_forest(
                g=g.copy(), patterns_df=patterns_df, articles_df=articles_df, field_name=field_name + "_",
                tag_profile=None, incorporate_ts_length_days=False, story_distance_resolution=None,
                importance_weighting_mode=importance_weighting_mode,
                layout_prog=None, smart_sizing=True, smart_shaping=True, smart_layout=True, remove_isolates_flag=False,
                timing_marks=timing_marks,
                save_image_path=str(Path(
                    "../interactive_summaries/%s_%s-tree_plain.html" % (story_id, field_name)
                ).resolve())
            )
            t_days_div = extract_and_plot_forest(
                g=g.copy(), patterns_df=patterns_df, articles_df=articles_df, field_name=field_name + "_",
                tag_profile=None, incorporate_ts_length_days=True, story_distance_resolution=None,
                importance_weighting_mode=importance_weighting_mode,
                layout_prog=None, smart_sizing=True, smart_shaping=True, smart_layout=True, remove_isolates_flag=False,
                timing_marks=timing_marks,
                save_image_path=str(
                    Path("../interactive_summaries/%s_%s-tree_temp.html" % (story_id, field_name)).resolve()
                )
            )
            t_tag_profile_div = extract_and_plot_forest(
                g=g.copy(), patterns_df=patterns_df, articles_df=articles_df, field_name=field_name + "_",
                tag_profile=tag_profile, incorporate_ts_length_days=False, story_distance_resolution=None,
                importance_weighting_mode=importance_weighting_mode,
                layout_prog=None, smart_sizing=True, smart_shaping=True, smart_layout=True, remove_isolates_flag=False,
                timing_marks=timing_marks,
                save_image_path=str(Path(
                    "../interactive_summaries/%s_%s-tree_query.html" % (story_id, field_name)
                ).resolve())
            )
            t_basic_div = extract_and_plot_forest(
                g=g.copy(), patterns_df=patterns_df, articles_df=articles_df, field_name=field_name + "_",
                tag_profile=tag_profile, incorporate_ts_length_days=True,
                story_distance_resolution=story_distance_resolution,
                importance_weighting_mode=importance_weighting_mode,
                layout_prog=None, smart_sizing=False, smart_shaping=True, smart_layout=True, remove_isolates_flag=False,
                timing_marks=timing_marks,
                save_image_path=str(Path(
                    "../interactive_summaries/%s_%s-tree_basic.html" % (story_id, field_name + "_")
                ).resolve())
            )
            t_simple_div = extract_and_plot_forest(
                g=g.copy(), patterns_df=patterns_df, articles_df=articles_df, field_name=field_name + "_",
                tag_profile=tag_profile, incorporate_ts_length_days=True,
                story_distance_resolution=story_distance_resolution,
                importance_weighting_mode=importance_weighting_mode,
                layout_prog=None, smart_sizing=True, smart_shaping=False, smart_layout=True, remove_isolates_flag=False,
                timing_marks=timing_marks,
                save_image_path=str(Path(
                    "../interactive_summaries/%s_%s-tree_simple.html" % (story_id, field_name)
                ).resolve())
            )
        t_div = extract_and_plot_forest(
            g=g.copy(), patterns_df=patterns_df, articles_df=articles_df, field_name=field_name + "_",
            tag_profile=tag_profile, incorporate_ts_length_days=True,
            story_distance_resolution=story_distance_resolution,
            importance_weighting_mode=importance_weighting_mode,
            layout_prog=None, smart_sizing=True, smart_shaping=True, smart_layout=True, remove_isolates_flag=False,
            timing_marks=timing_marks,
            save_image_path=str(
                Path("../interactive_summaries/%s_%s-tree.html" % (story_id, field_name)).resolve()
            )
        )

    return patterns_df, tag_support_div, timing_marks, g_div, t_plain_div, t_days_div, t_tag_profile_div, \
           t_basic_div, t_simple_div, t_div
# ---------------------------------------------- END THE MAIN FUNCTIONS ---------------------------------------------- #


########################################################################################################################
# ------------------------------------------------- PATTERN FUNCTION ------------------------------------------------- #
########################################################################################################################
def enrich_pattern_features(patterns_df, articles_df):
    # get article timestamps
    articles_df_reindexed = articles_df.set_index('id')
    # for 5.6M patterns and 250K articles, the following cell runs in ~3.5 minutes
    patterns_df['article_t'] = patterns_df['doc_ids'].map(
        lambda p_a_ids: [articles_df_reindexed.loc[a_id, 't'] for a_id in p_a_ids]
    )
    patterns_df['article_epochs'] = patterns_df['doc_ids'].map(
        lambda p_a_ids: [articles_df_reindexed.loc[a_id, 'epoch'] for a_id in p_a_ids]
    )
    del articles_df_reindexed
    # getting hashtag temporal features
    patterns_df['ts_start'] = patterns_df['article_t'].map(lambda ts: min(ts))
    patterns_df['ts_end'] = patterns_df['article_t'].map(lambda ts: max(ts))
    patterns_df['ts_start_epoch'] = patterns_df['article_epochs'].map(lambda ts: min(ts))
    patterns_df['ts_end_epoch'] = patterns_df['article_epochs'].map(lambda ts: max(ts))
    patterns_df['ts_length_days'] = (patterns_df['ts_end'] - patterns_df['ts_start']).map(lambda delta: delta.days)
    patterns_df['ts_n_dates'] = (
        patterns_df['ts_end'].map(lambda d: d.date()) - patterns_df['ts_start'].map(lambda d: d.date())
    ).map(lambda d: d.days + 1)
    return patterns_df
# ---------------------------------------------- END PATTERN FUNCTION ------------------------------------------------ #
