"""
after cloning the code one must first run 
git update-index --assume-unchanged Hashtagger/config.py 
to tell git not to track the changes to the config file and then put the config info

git ls-files -v|grep '^h'  # to get a list of dirs/files that are 'assume-unchanged'
git update-index --no-assume-unchanged Hashtagger/config.py  # to undo dirs/files that are set to assume-unchanged
"""


class Config:
    def __init__(self):
        # blacklists
        self.HASHTAG_BLACKLIST = [
            "#news", "#business", "#breaking", "#politics",
            "#jobs", "#world", "#rt", "#sport", "#breakingnews", "#follow", "#new",
            "#update", "#bbc"
        ]
        self.KEYWORD_BLACKLIST = [
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "BBC", "TheJournal", "ie", "Al", "Jazeera", "News", "CNN"
        ]

        # for generating and plotting stats
        self.PLOT_ARTICLE_HASHTAG_STATS_FLAG = False  # this is very slow for big article sets
        self.PLOT_TAG_PROFILE_STATS_FLAG = True
        self.PLOT_KEYWORD_PROFILE_STATS_FLAG = True
        self.PLOT_CORPUS_STATS_FLAG = False

        # for article relevance
        self.MIN_N_ARTICLES_FOR_ELBOW_CUT = 10  # default 1 # sets smaller than this will keep all their articles for summary extraction
        self.EXPORT_RELEVANT_ARTICLES_TO_CSV_FILENAME = "../data/%s_articles_summary.csv"  # %s is for the story id, export skipped if None

        # for SocialTree
        self.CREATE_SOCIALTREE_FLAG = True
        self.KEYWORD_EXTRACTION_UNIGRAM_MODE = None  # default: None # None means the 'keyword_list' will be constructe based on 'keywords' field, possible values: [None, 'random', 'equal', 'nouns_first', 'verbs_first', 'no_nouns', 'no_verbs'
        self.N_PROFILE_KEYWORDS = 5  # default: 5 # (ignored when self.KEYWORD_EXTRACTION_UNIGRAM_MODE=None) is the maximum number of keywords extracted from the pseudoarticle
        self.HASHTAGS_WEIGHT_IN_TAG_FIELD_SCORES = 0.75
        self.SOCIALTREE_TAG_PROFILE_FIELD = 'tag_profile'  # use 'hashtag_profile' with 'good_hashtags' to use hashtags only
        self.SOCIALTREE_TAG_LIST_FIELD = 'tags_list'  # use 'good_hashtags' with 'hashtag_profile' to use hashtags only
        self.TAG_REDUNDANCY_CONF_THRES = 100
        self.TAG_LEXICAL_REDUNDANCY_PREFIX = "#"
        self.TAG_LEXICAL_REDUNDANCY_SUFFIX = "$"
        self.MIN_TAG_PROFILE_SIZE = 7
        self.TAG_PATTERN_TYPE = 'frequent'  # takes values in ['frequent', 'closed', 'maximal', 'generators, 'rules']
        self.MIN_TAG_PATTERN_SUPPORT = -1
        self.TAG_IMPORTANCE_WEIGHTING_MODE = 'prod/diff'  # default: 'prod/diff' # takes values in ['sum', 'min', 'hmean', 'prod/diff']
        self.TAG_SUBSTORY_DISTANCE_RESOLUTION = 24 * 3600  # doesn't incorporate substory distances if None or 0
        self.VISUALIZE_INTERMEDIATE_TREES_FLAG = False

        # for KeywordTree
        self.CREATE_KEYWORDTREE_FLAG = True
        self.KEYWORDTREE_TAG_PROFILE_FIELD = 'keyword_profile'
        self.KEYWORDTREE_TAG_LIST_FIELD = 'keywords_list'
        self.KEYWORD_REDUNDANCY_CONF_THRES = 100
        self.KEYWORD_LEXICAL_REDUNDANCY_PREFIX = ""
        self.KEYWORD_LEXICAL_REDUNDANCY_SUFFIX = "$"
        self.MIN_KEYWORD_PROFILE_SIZE = 7
        self.KEYWORD_PATTERN_TYPE = 'frequent'  # takes values in ['frequent', 'closed', 'maximal', 'generators, 'rules']
        self.MIN_KEYWORD_PATTERN_SUPPORT = -2
        self.KEYWORD_IMPORTANCE_WEIGHTING_MODE = 'prod/diff'  # default: 'prod/diff' # takes values in ['sum', 'min', 'hmean', 'prod/diff']
        self.KEYWORD_SUBSTORY_DISTANCE_RESOLUTION = 24 * 3600  # doesn't incorporate substory distances if None or 0

        # for TFIDFTree (it uses the KeywordTree paramaters for the rest of the parameters)
        self.TFIDFTREE_TAG_PROFILE_FIELD = 'tfidf_profile'
        self.TFIDFTREE_TAG_LIST_FIELD = 'tfidf_keywords'

        # for wordcloud
        self.CREATE_WORDCLOUD_FLAG = True
        self.WORDCLOUD_CONTENT_FIELD_NAME = 'processed_pseudoarticle'
        self.WORDCLOUD_MAX_WORDS = 25

        # for KeyGraph
        self.CREATE_KEYGRAPH_FLAG = True
        self.KEYGRAPH_CONFFILE_NAME = "NewsConstants.txt"
        self.KEYGRAPH_CONTENT_FIELD_NAME = 'processed_pseudoarticle'

        # for MetroMap
        self.CREATE_METROMAP_FLAG = False
        self.METROMAP_CONTENT_FIELD_NAME = 'processed_pseudoarticle'

        # for StoryForest
        self.CREATE_STORYFOREST_FLAG = False
        self.STORYFOREST_CONTENT_FIELD_NAME = 'processed_pseudoarticle'


config = Config()
