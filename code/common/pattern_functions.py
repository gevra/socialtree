import logging
import pandas as pd
import numpy as np
import subprocess
import time
from pathlib import Path
from common.functions import split_list, match_all_in_list2df_col, merge_lists_with_common_items
from common.plotting_functions import plotly_hist_div

_logger = logging.getLogger(__name__)


def export_docs_for_eclat(list_of_docs, item_field_name, item_profile_list, collection_id, field_blacklist=None,
                          filepath=None):
    if field_blacklist is None:
        field_blacklist = []
    if filepath is None:
        filepath = Path("patterns")
    filepath.mkdir(exist_ok=True)
    infile_name = filepath.joinpath("story_doc_items_%s_%s.txt" % (collection_id, item_field_name))
    # output the document items to a file for Eclat to mine patterns of those
    with open(infile_name, 'w') as f:
        for a in list_of_docs:
            items_to_write = [
                t for t in a[item_field_name] if (t not in field_blacklist) and (t in item_profile_list)
            ]
            if len(items_to_write) > 0:
                f.write("%s\n" % (",".join(items_to_write)))
    return infile_name


def mine_patterns_with_eclat(eclat_path, infile_name, outfile_name, pattern_type, support=-3):
    pattern_type_options_dict = {"frequent": 's', "closed": 'c', "maximal": 'm', "generators": 'g', "rules": 'r'}
    # run Eclat
    eclat_result = subprocess.run(
        [
            eclat_path,
            '-t%s' % pattern_type_options_dict[pattern_type],
            '-C"%"',
            '-f","',  # field/item separators
            '-s%s' % support,
            '-v,%a,%i',  # the default is " (%S)" or " (%a)" if -s argument is negative
            '-k;',  # item separator for output
            infile_name,
            outfile_name
        ],
        # './eclat %s - C "%" - f "," - s%s %s %s' % (type_opt, support, infile_name, outfile_name),
        shell=False,  # if passing a single string, 'shell' must be True
        stdout=subprocess.PIPE
    )
    # read the patterns into a dataframe
    if eclat_result.returncode != 0:
        _logger.error("eclat execution failed!")
        return None
    patterns_df = read_borgelt_patterns(outfile_name, item_separator=";", field_separator=",")
    _logger.info(
        "there are %d patterns with support>=%s mined from the retrieved documents" % (patterns_df.shape[0], support)
    )
    return patterns_df


def mine_rules_with_eclat(eclat_path, infile_name, outfile_name, support=-3, min_conf=80, min_size=1, max_size=None):
    # extract the association rules
    eclat_rules_result = subprocess.run(
        [
            eclat_path,
            '-t%s' % 'r',
            '-C"%"',  # comment characters (default: "#")
            '-f","',
            '-s%s' % support,
            '-m%d' % min_size,  # minimum number of items per item set/association rule
            '-n%d' % max_size if max_size is not None else '',  # maximum number of items per item set/association rule
            '-c%d' % min_conf,  # minimum confidence of a rule as a percentage (default: 80)
            '-v,%a,%C,%i',  # the default is " (%X, %C)" or " (%a, %C)" if -s argument is negative
            '-o',  # use original definition of the support of a rule (body & head)
            infile_name,
            outfile_name
        ],
        # './eclat %s - C "%" - f "," - s%s %s %s' % (type_opt, support, infile_name, outfile_name),
        shell=False,  # if passing a single string, 'shell' must be True
        stdout=subprocess.PIPE
    )
    # read the rules into a dataframe
    if eclat_rules_result.returncode != 0:
        _logger.error("eclat execution for rule extraction failed!")
    rules_df = read_borgelt_rules(outfile_name)
    return rules_df


def read_borgelt_patterns(csv_file_name, item_separator=" ", field_separator=","):
    # the alphabetical sorting of the pattern elements is done to support
    # the exact match lookup on the patterns without e.g. making an 
    # assumption about their order
    patterns_df = pd.read_csv(csv_file_name, sep=field_separator, header=None, names=['pattern', 'support', 'size'])
    patterns_df['pattern_list'] = patterns_df['pattern'].map(
        # lambda x: sorted([p for p in x.strip().split(item_separator) if len(p) > 0])
        lambda x: sorted([p.replace(" ", "_") for p in x.strip().split(item_separator) if len(p) > 0])
    )
    """the above thing is only for the reason that the patterns are later indexed by the " "-separated  "pattern" field
    and ["abc", "def"] would be indistinguishable from ["abc def"] """
    patterns_df['pattern'] = patterns_df['pattern_list'].map(lambda x: " ".join(x))
    patterns_df[['support', 'size']] = patterns_df[['support', 'size']].astype(int)
    patterns_df['area'] = patterns_df[['support', 'size']].apply(lambda x: x[0] * x[1], axis=1)
    _logger.info("read the patterns from '%s' and produced a df of shape %s" % (csv_file_name, patterns_df.shape))
    return patterns_df


def read_borgelt_rules(csv_file_name):
    # the alphabetical sorting of the pattern elements is done to support
    # the exact match lookup on the patterns without e.g. making an
    # assumption about their order
    # in Borgelt's implementation, the head
    def parse_rule(rule_str):
        tail, head = rule_str.strip().split(" <- ")
        head = sorted(head.strip().split(" "))
        head_size = len(head)
        tail = sorted(tail.strip().split(" "))
        tail_size = len(tail)
        pattern_list = sorted(head + tail)
        pattern = " ".join(pattern_list)
        return head, head_size, tail, tail_size, pattern, pattern_list
    rules_df = pd.read_csv(csv_file_name, sep=",", header=None, names=['rule', 'support', 'conf', 'size'])
    rules_df['head'], rules_df['head_size'], rules_df['tail'], rules_df['tail_size'], rules_df['pattern'], \
        rules_df['pattern_list'] = zip(*rules_df['rule'].map(parse_rule))
    rules_df[['support', 'size']] = rules_df[['support', 'size']].astype(int)
    rules_df['area'] = rules_df[['support', 'size']].apply(lambda x: x[0] * x[1], axis=1)
    rules_df['conf'] = rules_df['conf'].astype(float)
    _logger.info("read the rules from '%s' and produced a df of shape %s" % (csv_file_name, rules_df.shape))
    return rules_df


def populate_pattern_support_doc_ids(patterns_df, docs_df, field_name):
    patterns_df.loc[patterns_df['size'] == 1, 'doc_ids'] = \
        patterns_df[patterns_df['size'] == 1].apply(
            get_pattern_support_doc_ids_from_docs_df,
            docs_df=docs_df, field_name=field_name,
            axis=1
        )
    tttt = time.time()
    for size in range(2, patterns_df['size'].max() + 1):
        patterns_df = get_superset_pattern_doc_ids(patterns_df, size)
        _logger.debug(
            "finished getting the document ids for patterns of length %d in %.2f sec" % (size, time.time() - tttt)
        )
        tttt = time.time()
    del tttt
    return patterns_df


def get_pattern_support_doc_ids_from_docs_df(p, docs_df, field_name):
    doc_ids = docs_df[docs_df[[field_name]].apply(
        func=match_all_in_list2df_col,
        list_of_tags=p['pattern_list'], field_name=field_name,
        axis=1
    )]['id'].tolist()
    return doc_ids


def lookup_in_singletons(p, singletons_df):
    return singletons_df.at[p['pattern'], 'doc_ids']


def get_support_from_subsets(p, patterns_df):
    subset1, subset2 = split_list(p['pattern_list'])
    subset1_doc_ids = patterns_df.at[" ".join(sorted(subset1)), 'doc_ids']
    subset2_doc_ids = patterns_df.at[" ".join(sorted(subset2)), 'doc_ids']
    try:
        doc_ids = list(set(subset1_doc_ids).intersection(subset2_doc_ids))
    except TypeError:
        """ this is here just for some weird bug , so it's not a part of the algo logic! 
        the problem was that some time at[] would return array([list([ 67857, 7989, 9870]), nan])"""
        _logger.warning("met a problem when getting the support for %s ... trying to fix ..." % p['pattern_list'])
        if isinstance(subset1_doc_ids, np.ndarray):
            subset1_doc_ids = subset1_doc_ids[0]
        if isinstance(subset2_doc_ids, np.ndarray):
            subset2_doc_ids = subset2_doc_ids[0]
        doc_ids = list(set(subset1_doc_ids).intersection(subset2_doc_ids))
        _logger.warning("seems like the problem of getting the support for %s got solved" % p['pattern_list'])
    # sanity check
    if p['support'] != len(doc_ids):
        # print("'%s' has support %d" % (" ".join(sorted(subset1)), len(subset1_doc_ids)))
        # print("'%s' has support %d" % (" ".join(sorted(subset2)), len(subset2_doc_ids)))
        # print("'%s' has support %d" % (p['pattern'], len(doc_ids)))
        raise Exception("wrong set of support documents for %s" % p['pattern'])
    """ sometimes a weird and not consistently reproducible bug will cause "australia cyclon debbie queensland" pattern
    to yield doc_ids = 450718 instead of [449985, 450691, 450696, 450699, 458865, 450718] """
    return doc_ids
    

def get_superset_pattern_doc_ids(patterns_df, size):
    """
    this function expects pattern_df to have document ids for sets of length <size>//2
    """
    # sanity check
    n_subset_nans = patterns_df[patterns_df['size'] == size//2]['doc_ids'].isnull().sum()
    if n_subset_nans > 0:
        raise Exception("there are %d subsets without doc_ids" % n_subset_nans)
    
    try:
        # for a weird bug https://stackoverflow.com/q/39623187/2262424
        patterns_df.loc[patterns_df['size'] == size, 'doc_ids'] = \
            patterns_df.loc[patterns_df['size'] == size].apply(
                get_support_from_subsets, 
                patterns_df=patterns_df[['pattern', 'doc_ids']].set_index('pattern'),
                axis=1
            )
    except:
        _logger.warning("a weird BUG spoiled the fun with patterns of size %d, now trying the loop" % size)
        patterns_df_reindexed = patterns_df[['pattern', 'doc_ids']].set_index('pattern')
        for i, p in patterns_df[patterns_df['size'] == size].iterrows():
            patterns_df.set_value(
                i, 
                'doc_ids',
                get_support_from_subsets(p=p, patterns_df=patterns_df_reindexed)
            )
    return patterns_df


def find_redundant_sets_from_association_rules(eclat_path, infile_name, rule_outfile_name,
                                               support=-1, redundancy_conf_thres=100, item_profile=None):
    # item_profile is used only for ordering the redundant items...
    # the lower weight items will be considered redundant to their higher weight counterparts
    rules_df = mine_rules_with_eclat(
        eclat_path=eclat_path,
        infile_name=infile_name, outfile_name=rule_outfile_name, support=support,
        min_conf=redundancy_conf_thres, min_size=2, max_size=2
    )

    # find the redundant items
    rules_df_clean = rules_df[(rules_df['conf'] >= redundancy_conf_thres) & (rules_df['size'] == 2)].copy()
    if item_profile is None:
        all_items = set(item for p in rules_df_clean['pattern_list'].tolist() for item in p)
        item_profile = dict(zip(all_items, [1] * len(all_items)))
    pattern_occurrence_count = rules_df_clean['pattern'].value_counts()
    pattern_occurrence_count_df = pd.DataFrame(
        {'pattern': pattern_occurrence_count.index, 'rule_count': pattern_occurrence_count.values}
    )
    # a set of size n can generate n! rules, however ECLAT by Borgelt only yields rules with 1 item in the tail
    # that means that maximum number of rules would be equal to the size of the set
    # BUT this will cause a problem as now, not all the subsets may yield rules even if the superset does
    pattern_occurrence_count_df['max_rule_count'] = pattern_occurrence_count_df['pattern'].map(
        lambda x: len(x.split(" "))  # would have been math.factorial(len(x.split(" ")))
    )
    # given the comments above, the co-occurrence can be safely defined only for pairs
    list_of_redundant_sets = pattern_occurrence_count_df[
        (pattern_occurrence_count_df['rule_count'] == pattern_occurrence_count_df['max_rule_count']) &
        (pattern_occurrence_count_df['rule_count'] == 2)
        ]['pattern'].map(lambda p: sorted(p.split(" "), key=lambda i: item_profile[i], reverse=True)).tolist()
    # now we need to make sure that we don't delete all the items of a redundant set,
    # as a n-set can be split to n*(n-1)/2 pairs and all n items may be left to be deleted, or given the sorted() above,
    # the most query-relevant (or the alphabetically first, if key=None) item will survive if left as is
    # Note! leaving key=None makes sense for the cases when matching of patterns needs to be done with another df

    # merge lists with common elements/items
    # this will also make sure that the items to be deleted, don't appear in among the keys of the dict, this may happen
    # because the sets of n redundant items will appear as n*(n-1)/2 pairs
    list_of_redundant_sets = merge_lists_with_common_items(list_of_redundant_sets)
    list_of_redundant_sets = [sorted(s, key=lambda i: item_profile[i], reverse=True) for s in list_of_redundant_sets]

    # create a dict where the keys are the items with biggest profile score in each of the redundant item sets
    # and the corresponding values are the remaining items of the redundant sets
    redundant_items_dict = dict([(s[0], s[1:]) for s in list_of_redundant_sets])
    items_to_remove = [i for s in redundant_items_dict.values() for i in s]
    _logger.info(
        "found %d redundant sets of items with %.f%% conditional probability of co-occurrence on both directions,"
        "\nas a result, %d items will be removed from the item profile "
        "and later will be added to their co-occurring items" %
        (len(list_of_redundant_sets), redundancy_conf_thres, len(items_to_remove))
    )
    return list_of_redundant_sets


def find_children(patterns_df):
    # currently written in a very inefficient way!!!
    # find and populate the pattern children column
    patterns_df_copy = patterns_df.copy()  # copy is created in order to not modify the df when iterating over it
    patterns_df["children_ids"] = patterns_df["pattern_list"]
    for (idx, p) in patterns_df_copy.iterrows():
        patterns_df.set_value(
            idx,
            "children_ids",
            [
                ii for (ii, pp) in patterns_df_copy[patterns_df_copy["size"] == p["size"] + 1].iterrows()
                if set(p["pattern_list"]).issubset(set(pp["pattern_list"])) and ii != idx
            ]
        )
    # create n_children column
    patterns_df["n_children"] = patterns_df["children_ids"].map(lambda x: len(x))
    return patterns_df


def plot_pattern_stats(patterns_df, field_name):
    list_of_x_name_opacity_tuples = [
        (
            patterns_df[patterns_df['size'] == 1]['support'].tolist(),
            "support",
            dict(
                start=patterns_df[patterns_df['size'] == 1]['support'].min() - 0.5,
                end=patterns_df[patterns_df['size'] == 1]['support'].max() + 0.5,
                size=1
            ),
            0.9
        )
    ]

    tag_support_div = plotly_hist_div(
        list_of_x_name_opacity_tuples,
        barmode=None,
        title="%s support distribution" % field_name, image_height=740, image_width=980, margin_dict=None,
        x_title="number of supporting documents", y_title="number of %ss" % field_name,
        showlegend=False, autorange=None,
        shapes_list=None, legend_dict=None, annotations_list=None, save_image_path=None
    )
    return tag_support_div
