import logging
import numpy as np
import time
import datetime
import inflect
from pathlib import Path
from collections import defaultdict, Counter
from common.functions import match_any_in_list2df_col, normalize_dict
from common.functions import find_curve_elbow_idx_based_on_max_dist
from common.plotting_functions import plotly_scatter_div, plotly_hist_div
from common.pattern_functions import export_docs_for_eclat, find_redundant_sets_from_association_rules

_p = inflect.engine()
_logger = logging.getLogger(__name__)


def get_story_tag_profile(articles_df, tag_field_name, tag_blacklist=None, normalized=True):
    if tag_blacklist is None:
        tag_blacklist = []
    unique_tags_tuples_weighted = defaultdict(float)
    sum_of_all_the_weights = 0
    for i, a in articles_df.iterrows():
        reweighted_tags_of_a = [
            (t, t_score * a['score'])
            for (t, t_score) in a[tag_field_name].items()
            if t not in tag_blacklist
        ]
        for t, t_weighted_score in reweighted_tags_of_a:
            unique_tags_tuples_weighted[t] += t_weighted_score
            sum_of_all_the_weights += t_weighted_score
    # normalize the scores
    if normalized:
        normalized_tag_profile = dict((t, s/sum_of_all_the_weights) for (t, s) in unique_tags_tuples_weighted.items())
        return normalized_tag_profile
    else:
        return dict(unique_tags_tuples_weighted)


def check_tag_profile_recall(articles_df, tag_profile, tag_field_name, sparsify_by=20):
    list_of_tags = [t for t, s in sorted(tag_profile.items(), key=lambda x: x[1], reverse=True)]
    recall_count = []
    for n_tags in range(1, len(list_of_tags) + 1, max(1, int(np.floor(len(list_of_tags)/sparsify_by)))):
        article_count = articles_df[articles_df[[tag_field_name]].apply(
            func=match_any_in_list2df_col,
            args=(list_of_tags[:n_tags], tag_field_name,),
            axis=1
        )].shape[0]
        recall_count.append((n_tags, article_count))
    return recall_count


def reduce_tag_profile(articles_df, tag_profile, tag_field_name, min_recall_thres=1, recall_count=None):
    """
    Return the smallest subset of tags with a recall > min_recall_thres.
    :param articles_df:
    :param tag_profile:
    :param tag_field_name:
    :param min_recall_thres:
    :param recall_count:
    :return:
    """
    if recall_count is None:
        recall_count = check_tag_profile_recall(articles_df, tag_profile, tag_field_name, sparsify_by=20)
    # get a rough guess where is the number of tags for getting the required recall
    idx_above_thres = np.where(
        np.array([article_count for n_tags, article_count in recall_count]) >= min_recall_thres * articles_df.shape[0]
    )[0]
    if len(idx_above_thres) == 0:
        _logger.warning("even the full set of tags doesn't reach the recall threshold of %.2f" % min_recall_thres)
        return tag_profile
    elif idx_above_thres[0] == 0:
        return dict(sorted(tag_profile.items(), key=lambda x: x[1], reverse=True)[0])
    else:
        # find finer area, where the recall hits the thres
        list_of_tags = [t for t, s in sorted(tag_profile.items(), key=lambda x: x[1], reverse=True)]
        for n_tags in range(recall_count[idx_above_thres[0] - 1][0], recall_count[idx_above_thres[0]][0] + 1):
            article_count = articles_df[articles_df[[tag_field_name]].apply(
                func=match_any_in_list2df_col,
                args=(list_of_tags[:n_tags], tag_field_name,),
                axis=1
            )].shape[0]
            if article_count >= min_recall_thres * articles_df.shape[0]:
                return dict(sorted(tag_profile.items(), key=lambda x: x[1], reverse=True)[:n_tags])


def plot_tag_profile_recall(recall_count):
    div = plotly_scatter_div(
        list_of_x_y_name_hovertext_tuples=[
            (
                [n_tags for n_tags, n_articles in recall_count],
                [n_articles for n_tags, n_articles in recall_count],
                "tag_recall",
                None
            )
        ],
        mode='lines+markers', title="recall count",
        image_height=740, image_width=980, x_title="tags", y_title="articles"
    )
    return div


def remove_lexical_duplicates(article_profile, set_of_tags, prefix_seq="$", suffix_seq="$"):
    """
    Note that the assumption is that all the keys of article_profile are found also in set_of_tags
    # TODO: add a redundancy check for named entities w.g. #trump=#donaldtrump (may require entity disambiguation)
    """
    problematic_nouns = ["news", "#news", "#series", "#species", "#congries", "#crossroads", "#headquarters"]
    article_profile_clean = {}
    for tag, tag_score in sorted(article_profile.items(), key=lambda x: x[1], reverse=True):
        # the sorting makes sure that the keywords (n-grams) are checked before hashtags
        if tag[0] == "#":
            is_hashtag = True
            other_type = tag[1:]
            prefix = "#"
        else:
            is_hashtag = False
            other_type = "#" + tag
            prefix = ""
        suffix = ""
        if other_type in set_of_tags:
            prefix = prefix_seq
        elif not is_hashtag and "_" in tag and "#" + tag.replace("_", "") in set_of_tags:
            # to match e.g. 'north_korea' and '#northkorea'
            prefix = prefix_seq
            tag = tag.replace("_", "")

        # check for singular-plural duplicates and if both exist append $ to the end of the singular
        singular_tag = _p.singular_noun(tag)  # will be set to False if tag was singular already
        if singular_tag:  # the tag is in plural
            singular_other_type = _p.singular_noun(other_type)
            if singular_tag in set_of_tags:
                # the tag was plural and the singular is there too
                suffix = suffix_seq
            if _p.singular_noun(other_type) in set_of_tags:
                # the tag was plural and the singular of the other type is there too
                suffix = suffix_seq
                prefix = prefix_seq
            if singular_tag == tag or tag in problematic_nouns:  # when the singular and plural have the same form!
                # this should work on words like "fish", "aircraft", "series" etc.
                # https://www.quora.com/For-which-English-nouns-are-the-singular-and-plural-forms-the-same-word
                suffix = ""
                singular_tag = tag  # this is needed cause _p.singular_noun("#series") = "#sery"
            processed_tag = prefix + (
                (singular_other_type if is_hashtag else singular_tag) + suffix if suffix == suffix_seq
                else (other_type if is_hashtag else tag)
            )
        else:  # the tag is singular
            if _p.plural_noun(tag) in set_of_tags:
                # the tag was singular and the plural is there too
                suffix = suffix_seq
            if _p.plural_noun(other_type) in set_of_tags:
                # the tag was singular and the plural of the other type is there too
                suffix = suffix_seq
                prefix = prefix_seq
            if _p.plural_noun(other_type) == other_type or tag in ["new", "news"]:
                # when the singular and plural have the same form!
                # this should work on words like "#fish", "#aircraft" etc. cause inflect sees these singular
                suffix = ""
            processed_tag = prefix + (other_type if is_hashtag else tag) + suffix

        if processed_tag in article_profile_clean:
            article_profile_clean[processed_tag] += tag_score
        else:
            article_profile_clean[processed_tag] = tag_score
    # to support the merging of the n-grams and the hashtags (e.g. 'north_korea' and '#northkorea')
    article_profile_recleaned = {}
    for clean_tag, clean_tag_score in article_profile_clean.items():
        if clean_tag[0] == "#" and prefix_seq + clean_tag[1:] in article_profile_clean:
            recleaned_tag = prefix_seq + clean_tag[1:]
        else:
            recleaned_tag = clean_tag
        if recleaned_tag in article_profile_recleaned:
            article_profile_recleaned[recleaned_tag] += clean_tag_score
        else:
            article_profile_recleaned[recleaned_tag] = clean_tag_score
    return article_profile_recleaned


def get_reduced_tag_profile_and_stats(articles_df, tag_profile_field_name, tag_blacklist, story_id="0",
                                      timing_marks=None,
                                      min_article_recall=None, redundancy_conf_thres=100, min_profile_size=None,
                                      lexical_redundancy_prefix="", lexical_redundancy_suffix="",
                                      normalized=True, plot_stats_flag=False):
    # ---------------------------------------------------- FILTER 0 ----------------------------------------------------
    # remove/replace the lexically redundant tags
    _logger.info("started replacing the lexiacally redundant tags")
    """ Because the patterns are computed after the redundancy removal and because of the way the pattern support set is
        selected ('redundant_tags_dict' is used only in the visualization and does not affect support calculation),
        it makes sense to create a new redundancy-free field without distorting the real tag_profile values.
        E.g. 'tag_profile'={"impeachment": 0.25, "#trump": 0.4, "#impeachment": 0.15, "#flood": 0.1, "floods": 0.1}
        will be stored in 'tag_profile_original' and will be replaced with 
        'tag_profile'={"$impeachment": 0.4, "#trump": 0.6, "$flood$": 0.2}. The 'tags_list' must be changed accordingly.
        The only reason for keeping the original tag_profile and tags_list 
        is to be able to distinguish the hashtags that are indeed used on social media from keywords."""
    articles_df[tag_profile_field_name + "_original"] = articles_df[tag_profile_field_name]
    set_of_tags = set(tag for a_tags in articles_df[tag_profile_field_name].tolist() for tag in a_tags)
    articles_df[tag_profile_field_name] = articles_df[tag_profile_field_name].apply(
        remove_lexical_duplicates, set_of_tags=set_of_tags,
        prefix_seq=lexical_redundancy_prefix, suffix_seq=lexical_redundancy_suffix
    )
    # to support the merging of the n-grams and the hashtags (e.g. 'north_korea' and '#northkorea')
    set_of_deduplicated_tags = set(tag for a_tags in articles_df[tag_profile_field_name].tolist() for tag in a_tags)
    articles_df[tag_profile_field_name] = articles_df[tag_profile_field_name].map(
        lambda article_tag_profile: dict([
            (lexical_redundancy_prefix + tag[1:], tag_score) if tag[0] == "#" and lexical_redundancy_prefix + tag[1:]
            in set_of_deduplicated_tags
            else (tag, tag_score)
            for (tag, tag_score) in article_tag_profile.items()
        ])
    )

    # ----------------------------------------------------- STATS ------------------------------------------------------
    _logger.info("--- started the story tag profile extraction ---")
    tag_profile = get_story_tag_profile(
        articles_df[['score', tag_profile_field_name]],
        tag_field_name=tag_profile_field_name,
        tag_blacklist=tag_blacklist,
        normalized=normalized
    )
    # tag_profile_original = get_story_tag_profile(
    #     articles_df[['score', tag_profile_field_name + "_original"]],
    #     tag_field_name=tag_profile_field_name + "_original",
    #     tag_blacklist=tag_blacklist,
    #     normalized=normalized
    # )
    # n_t = 20
    # _logger.debug(
    #     "before the lexical redundancy removal there were %d tags \ntop %d being: %s \nnow the top %d are: %s" % (
    #         len(tag_profile_original), n_t,
    #         ["%s:%.3f" % ts for ts in sorted(tag_profile_original.items(), key=lambda x: x[1], reverse=True)[:n_t]],
    #         n_t,
    #         ["%s:%.3f" % ts for ts in sorted(tag_profile.items(), key=lambda x: x[1], reverse=True)[:n_t]]
    #     )
    # )
    # Note that the articles which have havehashtag=True
    # may have no good enough hashtags and therefore hashtag_profile={}
    _logger.info(
        "initially there were %d tags selected for story's profile %s" % (len(tag_profile), tag_profile_field_name)
    )

    # ---------------------------------------------------- FILTER 1 ----------------------------------------------------
    # remove the fully redundant tags based on the extracted high-confidence association rules of tag co-occurrence
    infile_name = export_docs_for_eclat(
        list_of_docs=articles_df.to_dict('records'), item_field_name=tag_profile_field_name,
        item_profile_list=[t for t, score in tag_profile.items()], collection_id=story_id, field_blacklist=None
    )
    rule_outfile_name = Path("patterns/story_patterns_%s_%s_%s.out" % (story_id, tag_profile_field_name, "rules"))
    # item_profile is used only for ordering the redundant items...
    # the lower weight items will be considered redundant to their higher weight counterparts
    list_of_redundant_sets = find_redundant_sets_from_association_rules(
        eclat_path=Path('eclat').resolve(),
        infile_name=infile_name, rule_outfile_name=rule_outfile_name,
        support=-1, redundancy_conf_thres=redundancy_conf_thres, item_profile=tag_profile
    )
    list_of_redundant_sets = [sorted(s, key=lambda t: tag_profile[t], reverse=True) for s in list_of_redundant_sets]
    # create a dict where the keys are the tags with biggest profile score in each of the redundant tag sets
    # and the corresponding values are the remaining items of the redundant sets
    redundant_tags_dict = dict([(s[0], s[1:]) for s in list_of_redundant_sets])
    # Note that a tag can be redundant for 2 different tags, so tags_to_remove may contain multiple occurrences of a tag
    tags_to_remove = [t for s in redundant_tags_dict.values() for t in s]
    tags_to_remove_unique_count = Counter(tags_to_remove)
    # one option is to re-normalize the profile after the redundancy removal
    # tag_profile = dict([(t, score) for (t, score) in tag_profile.items() if t not in tags_to_remove])
    # re-normalize the tag profile
    # tag_profile = normalize_dict(tag_profile)
    # another option is to add the weights of redundant tags to the remaining tag,
    # thus not disturbing the topical distribution balance
    for tag_set in list_of_redundant_sets:
        tag_profile[tag_set[0]] += sum([tag_profile[t] / tags_to_remove_unique_count[t] for t in tag_set[1:]])
    tag_profile = dict([(t, score) for (t, score) in tag_profile.items() if t not in tags_to_remove])
    # sanity check
    if abs(sum(tag_profile.values()) - 1) > 1e-6:
        _logger.warning(
            "!!! the sum of the tag scores after the redundancy removal is %.4f (must be 1)" % sum(tag_profile.values())
        )
    _logger.info(
        "after the redundancy removal, %d tags compose story's profile %s" % (len(tag_profile), tag_profile_field_name)
    )

    # ----------------------------------------------------- STATS ------------------------------------------------------
    n_articles = articles_df.shape[0]
    # plot the tag score distribution
    tag_profile_list = sorted(tag_profile.items(), key=lambda x: x[1], reverse=True)
    max_tag_score = max([s for (t, s) in tag_profile_list])

    cutoff_index_max_dist = find_curve_elbow_idx_based_on_max_dist([s for (t, s) in tag_profile_list])
    _logger.info("the cuttoff point should be %s based on max dist" % str(tag_profile_list[cutoff_index_max_dist]))

    if plot_stats_flag:
        tag_profile_div = plotly_scatter_div(
            list_of_x_y_name_hovertext_tuples=[
                (
                    [s / max_tag_score * 100 for (t, s) in tag_profile_list],
                    list(range(1, len(tag_profile_list) + 1)),
                    "tags",
                    ["%s: %.5f" % (t, s) for (t, s) in tag_profile_list]
                )
            ],
            mode='lines+markers', title="", autorange='reversed',
            image_height=330, image_width=700, x_title="tag profile score cutoff (%)", y_title="", showlegend=False,
            margin_dict={'l': 40, 'r': 5, 'b': 35, 't': 30, 'pad': 4},
            shapes_list=[
                # vertical line at the elbow
                {
                    'type': 'line', 'xref': 'x', 'yref': 'paper',
                    'x0': tag_profile_list[cutoff_index_max_dist][1] / max_tag_score * 100, 'y0': 0,
                    'x1': tag_profile_list[cutoff_index_max_dist][1] / max_tag_score * 100, 'y1': 1,
                    'line': {'color': 'rgb(44, 160, 44)', 'width': 3, 'dash': 'dot'}
                },
                # vertical line at 1/n_articles
                {
                    'type': 'line', 'xref': 'x', 'yref': 'paper',
                    'x0': 1 / n_articles / max_tag_score * 100, 'y0': 0,
                    'x1': 1 / n_articles / max_tag_score * 100, 'y1': 1,
                    'line': {'color': 'rgb(255, 127, 14)', 'width': 2, 'dash': 'dot'}
                },
                # highlight from elbow cutoff - 0% of max
                {
                    'type': 'rect', 'xref': 'x', 'yref': 'paper',
                    'x0': tag_profile_list[cutoff_index_max_dist][1] / max_tag_score * 100, 'y0': 0, 'x1': 0, 'y1': 1,
                    'opacity': 1, 'fillcolor': '#E2E2E2',  # '#969696',  # '#d3d3d3',
                    'line': {'width': 0}, 'layer': 'below'
                }
            ],
            annotations_list=[
                {
                    'x': tag_profile_list[cutoff_index_max_dist][1] / max_tag_score * 100 + 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'text': 'elbow cutoff at %d%% score drop' %
                            int((1 - tag_profile_list[cutoff_index_max_dist][1] / max_tag_score) * 100),
                    'textangle': -90,
                    'xref': 'x',
                    'yref': 'paper',
                    'bgcolor': 'rgba(0,0,0,0)',  # '#E2E2E2',
                },
                {
                    'x': 1 / n_articles / max_tag_score * 100 + 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'text': '1/%d = %.4f = %.1f%% score drop' % (
                        n_articles, 1 / n_articles, (1 - 1 / n_articles / max_tag_score) * 100
                    ),
                    'textangle': -90,
                    'xref': 'x',
                    'yref': 'paper',
                    'bgcolor': 'rgba(0,0,0,0)',  # '#E2E2E2',
                }
            ],
            save_image_path=str(Path(
                "../interactive_summaries/%s_%s_hashtag_rel_cutoff.html" % (story_id, tag_profile_field_name)
            ).resolve())
        )
    else:
        tag_profile_div = ""

    # calculate profile's recall
    tag_profile_recall = articles_df[articles_df[[tag_profile_field_name]].apply(
        match_any_in_list2df_col,
        list_of_tags=[t for (t, s) in tag_profile_list if s >= tag_profile_list[cutoff_index_max_dist][1]],
        field_name=tag_profile_field_name,
        axis=1
    )].shape[0]

    if plot_stats_flag:
        tag_profile_div = "<h4><strong>%d</strong> tags selected from %d, <strong>%.2f</strong> cumulative weight, " \
                          "<strong>%.2f%%</strong> recall</h4>" % (
                              cutoff_index_max_dist + 1, len(tag_profile_list),
                              sum(s for (t, s) in tag_profile_list if s >= tag_profile_list[cutoff_index_max_dist][1]),
                              tag_profile_recall / articles_df.shape[0] * 100
                          ) + tag_profile_div

    # ---------------------------------------------------- FILTER 2 ----------------------------------------------------
    # remove the non-relevant tags from the profile
    if min_profile_size is not None:
        if min_profile_size > cutoff_index_max_dist + 1:
            _logger.info(
                "the profile size was %d, but was forced to be >=%d" % (cutoff_index_max_dist, min_profile_size)
            )
            forcefully_added_tags = tag_profile_list[cutoff_index_max_dist + 1: min_profile_size]
            cutoff_index_max_dist = max(cutoff_index_max_dist, min_profile_size - 1)
            # a quick and dirty fix for cases when cutoff_index_max_dist is bigger than the tag profile length
            cutoff_index_max_dist = min(cutoff_index_max_dist, len(tag_profile_list) - 1)

            # calculate profile's recall
            tag_profile_recall_new = articles_df[articles_df[[tag_profile_field_name]].apply(
                func=match_any_in_list2df_col,
                args=(
                    [t for (t, s) in tag_profile_list if s >= tag_profile_list[cutoff_index_max_dist][1]],
                    tag_profile_field_name,
                ),
                axis=1
            )].shape[0]
            if plot_stats_flag:
                tag_profile_div = "after adding the following tags: %s" \
                                  "<h4><strong>%d</strong> tags selected from %d, " \
                                  "<strong>%.2f</strong> cumulative weight, <strong>%.2f%%</strong> recall</h4> " \
                                  "before was: " % (
                                      ["%s: %.2f" % (t, s) for (t, s) in forcefully_added_tags],
                                      cutoff_index_max_dist + 1, len(tag_profile_list),
                                      sum(s for (t, s) in tag_profile_list if s >= tag_profile_list[cutoff_index_max_dist][1]),
                                      tag_profile_recall_new / articles_df.shape[0] * 100
                                  ) + tag_profile_div
    tag_profile = dict([
        (t, score) for (t, score) in tag_profile.items() if score >= tag_profile_list[cutoff_index_max_dist][1]
    ])
    # re-normalize the tag profile
    # tag_profile = normalize_dict(tag_profile)
    # _logger.info(
    #     "after the irrelevant tags with score<%.4f (corresponds to 1/%d, i.e. 1/N_articles) are removed, "
    #     "only %d tags compose story's %s profile" %
    #     (1 / n_articles, n_articles, len(tag_profile.keys()), tag_profile_field_name)
    # )

    if timing_marks is not None:
        _logger.info(
            "--- finished the story %s profile extraction in %.3f seconds (%.1f seconds since start) ---" %
            (tag_profile_field_name, time.time() - timing_marks[-1][1], time.time() - timing_marks[0][1])
        )
        timing_marks.append(('extract_story_%s_profile' % tag_profile_field_name, time.time()))

    # ---------------------------------------------------- FILTER 3 ----------------------------------------------------
    # plot the tag_profile recall
    tag_recall_count = check_tag_profile_recall(
        articles_df=articles_df[['id', tag_profile_field_name]],
        tag_profile=tag_profile,
        tag_field_name=tag_profile_field_name,
        sparsify_by=20
    )
    if plot_stats_flag:
        tag_recall_div = plot_tag_profile_recall(tag_recall_count)
    else:
        tag_recall_div = ""
    if timing_marks is not None:
        _logger.info(
            "--- finished the %s plot preparations in %.3f seconds (%.1f seconds since start) ---" %
            (tag_profile_field_name, time.time() - timing_marks[-1][1], time.time() - timing_marks[0][1])
        )
        timing_marks.append(('plot_%s_stats' % tag_profile_field_name, time.time()))

    if min_article_recall is not None:
        # reduce the tag profile
        n_profile_tags = len(tag_profile.items())
        tag_profile = reduce_tag_profile(
                articles_df=articles_df[['id', tag_profile_field_name]],
                tag_profile=tag_profile,
                tag_field_name=tag_profile_field_name,
                min_recall_thres=min_article_recall,
                recall_count=tag_recall_count
            )
        _logger.info(
            "reduced the number of %ss in the profile from %d to %d to achieve %.2f%% recall" %
            (tag_profile_field_name.split("_")[0], n_profile_tags, len(tag_profile.items()), min_article_recall * 100)
        )
        if plot_stats_flag:
            tag_recall_div = "<br><p>reduced the number of %ss in the profile from %d to %d for >=%.2f%% recall</p>" % \
                             (
                                 tag_profile_field_name.split("_")[0], n_profile_tags,
                                 len(tag_profile.items()), min_article_recall * 100
                             ) + tag_recall_div
        # renormalize the profile
        if normalized:
            tag_profile = normalize_dict(tag_profile)
        if timing_marks is not None:
            _logger.info(
                "--- finished the %s reduction in %.3f seconds (%.1f seconds since start) ---" %
                (tag_profile_field_name, time.time() - timing_marks[-1][1], time.time() - timing_marks[0][1])
            )
            timing_marks.append(('reduce_story_%s' % tag_profile_field_name, time.time()))
    return tag_profile, redundant_tags_dict, tag_recall_div, tag_profile_div, timing_marks


def count_unique_tags(articles_df, field_name):
    set_of_unique_tags = set([])
    n_unique_tags = []
    for i, a in articles_df.iterrows():
        set_of_unique_tags |= set(a[field_name])
        n_unique_tags.append(len(set_of_unique_tags))
    return n_unique_tags


def count_occuring_tags(articles_df, field_name):
    n_tags = []
    for i, a in articles_df.iterrows():
        n_tags.append(len(a[field_name]))
    return np.cumsum(n_tags)


def determine_optimal_article_cutoff(query):
    len_all_query_terms = len(query.split())
    if len_all_query_terms > 1:
        score_drop_coeff = 1 - 1.0 / len_all_query_terms
    else:
        score_drop_coeff = 0.5
    _logger.info("--- score_drop_coeff=%.2f ---" % score_drop_coeff)

    return score_drop_coeff


def plot_article_hashtag_stats(articles_df, field_name, hist_step=7*24*3600, article_rel_cutoff=0.1):
    # sort articles by their relevance
    articles_df.sort_values(by=['score'], ascending=False, inplace=True)

    # unique tags count
    unique_tag_count = count_unique_tags(articles_df.sort_values(by=['score'], ascending=False), field_name)

    # plot the n_hashtag-n_article plot
    tag_article_div = plotly_scatter_div(
        list_of_x_y_name_hovertext_tuples=[
            (
                list(range(1, articles_df.shape[0] + 1, 1)),
                unique_tag_count,
                "tags",
                ["relevance: %.0f%%" % rel for rel in articles_df['score'].tolist() / articles_df['score'].max() * 100]
            )
        ],
        list_of_x_y_name_hovertext_tuples_y2=[
            (
                list(range(1, articles_df.shape[0] + 1, 1)),
                articles_df.sort_values(by=['score'], ascending=False)['score'].tolist() / articles_df['score'].max() * 100,
                "article relevance",
                ["unique tags: %d" % n_tags for n_tags in unique_tag_count]
            )
        ],
        mode='lines+markers', title="unique tag count in articles of decreasing relevance",
        image_height=740, image_width=1600,
        x_title="top articles", y_title="unique tags", y2_title="relevance score (%)", showlegend=True,
        legend_dict=dict(
            x=0.5, y=0.95, traceorder='normal', orientation='h', xanchor='center',
            font=dict(family='sans-serif', size=14, color='#000'),
            # bgcolor='#E2E2E2', bordercolor='#FFFFFF', borderwidth=2,
        ),
    )

    start_epoch = int(time.mktime(datetime.datetime.fromtimestamp(articles_df['epoch'].min()).date().timetuple()))
    end_epoch = int(time.mktime(datetime.datetime.fromtimestamp(articles_df['epoch'].max() + 86400).date().timetuple()))
    article_date_hist, bin_edges = np.histogram(
        articles_df['epoch'], bins=range(start_epoch, end_epoch + hist_step + 1, hist_step)
    )

    set_of_all_tags_to_date = set([])
    n_new_tags_by_date = np.zeros(len(bin_edges) - 1)
    for i, a in articles_df.sort_values(by=['t'], ascending=True).iterrows():
        n_new_tags_by_date[np.digitize(a['epoch'], bin_edges) - 1] += len(set(a[field_name]) - set_of_all_tags_to_date)
        set_of_all_tags_to_date |= set(a[field_name])

    # the route with pandas Timestamp type is PAINFUL and the good old UNIX epochs are the best
    # article_date_hist = articles_df['datetime'].groupby(
    #     [articles_df['datetime'].dt.year, articles_df['datetime'].dt.month, articles_df['datetime'].dt.day]
    # ).count().rename_axis(['year', 'month', 'day']).reset_index(name='daily_count', level=['year', 'month', 'day'])
    # article_date_hist['date'] = article_date_hist.apply(
    #     lambda x: datetime.datetime(x['year'], x['month'], x['day']), axis=1
    # )

    article_density_div = plotly_scatter_div(
        list_of_x_y_name_hovertext_tuples=[
            # (
            #     [datetime.datetime.fromtimestamp(e) for e in (bin_edges[:-1] + bin_edges[1:]) / 2],
            #     article_date_hist,
            #     "new articles by date",
            #     None
            # ),
            (
                [datetime.datetime.fromtimestamp(e) for e in (bin_edges[:-1] + bin_edges[1:]) / 2],
                np.cumsum(article_date_hist),
                "articles to date",
                None
            )
        ],
        list_of_x_y_name_hovertext_tuples_y2=[
            (
                [datetime.datetime.fromtimestamp(e) for e in (bin_edges[:-1] + bin_edges[1:]) / 2],
                np.cumsum(n_new_tags_by_date),
                "tags to date",
                None
            ),
            # (
            #     [datetime.datetime.fromtimestamp(e) for e in (bin_edges[:-1] + bin_edges[1:]) / 2],
            #     n_new_tags_by_date,
            #     "new tags by date",
            #     None
            # )
        ],
        mode='lines+markers', title="articles and tags per each %d day(s)" % (hist_step / (24 * 3600)),
        image_height=740, image_width=1600, x_title="date", y_title="articles", y2_title="tags", showlegend=True,
        legend_dict=dict(
            x=0.5, y=0.95, traceorder='normal', orientation='h', xanchor='center',
            font=dict(family='sans-serif', size=14, color='#000'),
            # bgcolor='#E2E2E2', bordercolor='#FFFFFF', borderwidth=2,
        ),
    )

    recall_relevance_div = plotly_scatter_div(
        list_of_x_y_name_hovertext_tuples=[
            (
                articles_df['score'].tolist() / articles_df['score'].max() * 100,
                [articles_df[articles_df['score'] >= thres].shape[0] for thres in articles_df['score']],
                "articles",
                ["unique tags: %d" % n_tags for n_tags in unique_tag_count]
            ),
            (
                articles_df['score'].tolist() / articles_df['score'].max() * 100,
                unique_tag_count,
                "tags",
                ["articles: %d" % n_articles for n_articles in
                 [articles_df[articles_df['score'] >= thres].shape[0] for thres in articles_df['score']]]
            )
        ],
        mode='lines+markers', title="", autorange='reversed',
        image_height=250, image_width=700, x_title="article relevance (% of max)", y_title="", showlegend=True,
        margin_dict={'l': 40, 'r': 5, 'b': 35, 't': 30, 'pad': 4},
        shapes_list=[
            # vertical line at 10%
            {
                'type': 'line', 'xref': 'x', 'yref': 'paper',
                'x0': article_rel_cutoff * 100, 'y0': 0, 'x1': article_rel_cutoff * 100, 'y1': 1,
                'line': {'color': 'rgb(55, 128, 191)', 'width': 3, 'dash': 'dot'}
            },
            # highlight from 10%-0%
            {
                'type': 'rect', 'xref': 'x', 'yref': 'paper',
                'x0': article_rel_cutoff * 100, 'y0': 0, 'x1': 0, 'y1': 1,
                'opacity': 1, 'fillcolor': '#E2E2E2',  # '#969696',  # '#d3d3d3',
                'line': {'width': 0}, 'layer': 'below'
            }
        ],
        legend_dict=dict(
            x=0.5, y=0.95, traceorder='normal', orientation='h', xanchor='center',
            font=dict(family='sans-serif', size=14, color='#000'),
            # bgcolor='#E2E2E2', bordercolor='#FFFFFF', borderwidth=2,
        ),
        annotations_list=[{
            'x': article_rel_cutoff * 100 + 2,
            'y': 0.6,
            'showarrow': False,
            # 'text': 'cutoff at %d%% relevance drop' % int((1 - article_rel_cutoff) * 100),
            'text': 'cutoff at the elbow',
            'textangle': -90,
            'font': dict(size=16),
            'xref': 'x',
            'yref': 'paper',
            'bgcolor': 'rgba(0,0,0,0)'  # '#E2E2E2',
        }],
        save_image_path=Path("../interactive_summaries/article_rel_cutoff.html").resolve()
    )

    list_of_x_name_xbins_opacity_tuples = [
        (
            articles_df['n_good_hashtags'].tolist(),
            "hashtag count",
            dict(start=articles_df['n_good_hashtags'].min()-0.5, end=articles_df['n_good_hashtags'].max()+0.5, size=1),
            0.9
        )
    ]

    article_hashtag_count_div = plotly_hist_div(
        list_of_x_name_xbins_opacity_tuples,
        barmode=None,
        title="Per-article hashtag count distribution", image_height=740, image_width=980, margin_dict=None,
        x_title="number of \"good\" hashtags assigned to an article", y_title="number of articles",
        showlegend=False, autorange=None,
        shapes_list=None, legend_dict=None, annotations_list=None, save_image_path=None
    )

    # NOTE! the following are not the isolates, because some other articles may still contain these tags!
    # clean_articles_df = articles_df[articles_df['score'] >= article_rel_cutoff * articles_df['score'].max()]
    # isolate_tag_article_counts = clean_articles_df[clean_articles_df['n_good_hashtags'] == 1]['good_hashtags']\
    #     .map(lambda x: x[0]).value_counts()
    # list_of_x_name_opacity_tuples = [
    #     (
    #         isolate_tag_article_counts.tolist(),
    #         "isolate tag article count",
    #         dict(
    #             start=isolate_tag_article_counts.min() - 0.5,
    #             end=isolate_tag_article_counts.max() + 0.5,
    #             size=1
    #         ),
    #         0.9
    #     )
    # ]
    # isolate_tag_article_counts_div = plotly_hist_div(
    #     list_of_x_name_opacity_tuples,
    #     barmode=None,
    #     title="Isolate tag article count", image_height=740, image_width=980, margin_dict=None,
    #     x_title="number of supporting articles", y_title="number of hashtags",
    #     showlegend=False, autorange=None,
    #     shapes_list=None, legend_dict=None, annotations_list=None, save_image_path=None
    # )
    # print(isolate_tag_article_counts)

    return tag_article_div, article_density_div, recall_relevance_div, article_hashtag_count_div


def plot_tag_stats(articles_df, list_of_field_names):
    tttt = time.time()
    tag_stats_div = plotly_scatter_div(
        list_of_x_y_name_hovertext_tuples=[
            (
                count_occuring_tags(articles_df.sort_values(by=['epoch'], ascending=True), field_name),
                count_unique_tags(articles_df.sort_values(by=['epoch'], ascending=True), field_name),
                field_name,
                [field_name] * articles_df.shape[0]
            ) for field_name in list_of_field_names
        ],
        mode='lines+markers', title="",
        image_height=330, image_width=700, x_title="tag count", y_title="unique tag count", showlegend=True,
        margin_dict={'l': 40, 'r': 5, 'b': 35, 't': 30, 'pad': 4},
        legend_dict=dict(
            x=0.5, y=0.95, traceorder='normal', orientation='h', xanchor='center',
            font=dict(family='sans-serif', size=14, color='#000'),
            # bgcolor='#E2E2E2', bordercolor='#FFFFFF', borderwidth=2,
        ),
        save_image_path=Path("../interactive_summaries/tag_stats.html").resolve()
    )
    _logger.info(
        "finished plotting tag stats for %s, which took %.4f seconds" % (str(list_of_field_names), time.time() - tttt)
    )

    return tag_stats_div
