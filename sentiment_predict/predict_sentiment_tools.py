import math
import time
import praw
from datetime import datetime

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from .sentiment_tools_variables import *
from praw.models.reddit.submission import Submission

sia = SentimentIntensityAnalyzer()
logger = logging.getLogger(__name__)

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent="myapp v1.0",
)


def to_number(value):
    try:
        num = float(value)
        if math.isnan(num) or num <= 0:
            return 1
    except (ValueError, TypeError):
        return 1


def get_targets_posts(crypto: str):
    """
    :param crypto: case-insensitive crypto to be analyzed (e.g. 'btc')
    :return: list of related posts
    """
    logger.debug("Fetching Posts...")
    posts = list(reddit.subreddit(crypto.lower()).new(limit=None))
    logger.debug("Posts fetched successfully!!!")
    return posts


def calc_age_score(created_utc, current_time):
    created_utc = created_utc or (current_time - MAX_NUMBERS['max_age'])
    if (current_time - created_utc) < 0:
        age_score = 0
    else:
        age_score = max(1 - (current_time - created_utc) / MAX_NUMBERS['max_age'], 0)

    return age_score


def normalize_scores(score, num_comments, view_counts) -> dict:
    return {
        "score": (score or 1) / MAX_NUMBERS['max_score'],
        "num_comments": min((num_comments or 1) / MAX_NUMBERS['max_comments'], 1),
        "view_count": min((view_counts or 1) / MAX_NUMBERS['max_view_count'], 1),
    }


def calc_impact_score(age_score, scores):
    impact_score = SCORE_WEIGHTS["age"] * age_score * sum(
        SCORE_WEIGHTS[key] * scores[key] for key in scores
    )

    return SCORE_WEIGHTS["base"] * impact_score


def extract_data_from_posts(post: dict, current_time=int(time.time())) -> dict | None:
    age_score = calc_age_score(post['created_utc'], current_time)
    # print(f"\n{post.get('timestamp', 'Not Test2')}")
    # print(age_score)
    if not age_score: return None
    scores = normalize_scores(to_number(post.get('score', 1)),
                              to_number(post.get('num_comments', 1)),
                              to_number(post.get('ups', 1) + post.get('downs', 0)))
    # print(scores)
    # print(type(post.get('num_comments')))
    if not any(scores.values()): return None
    impact_score = calc_impact_score(age_score, scores)
    # print(impact_score)

    return {
        'impact_score': impact_score,
        'url': post['url'],
        'created_utc': post['created_utc'],
        'date': datetime.utcfromtimestamp(post['created_utc']).strftime('%Y-%m-%d %H:%M:%S'),
        'title': post['title'],
        'selftext': post['selftext'],
        'score': scores['score'],
        'num_comments': scores['num_comments'],
        'view_count': scores['view_count'],
    }


def show_post_info(post_dict):
    print(f"Title: {post_dict['title']}, URL: {post_dict['url']}")
    print(
        f"Date: {post_dict['date']} Score: {post_dict['score']}, num_comments: {post_dict['num_comments']}, "
        f"view_count: {post_dict['view_count']}, Impact score: {post_dict['impact_score']}")


def get_text_from_post(post: dict):
    return str(post['title']) + ":: " + str(post['selftext'])


def keyword_filter(text, labels):
    return any(label.lower() in text.lower() for label in labels)


def analyze_sentiment_now(lst, timestamp=int(time.time())):
    if type(lst[0]) is Submission:
        data = [result for item in lst if (result := extract_data_from_posts(item.__dict__, timestamp)) is not None]
    else:
        data = [result for item in lst if (result := extract_data_from_posts(item, timestamp)) is not None]
    return calc_average_score(data)


def test_a_time(df, timestamp):
    data = [result for idx, row in df.iterrows() if
            (result := extract_data_from_posts(row.to_dict(), timestamp)) is not None]
    return calc_average_score(data)


def calc_average_score(result_lst, crypto_symbol=None):
    if crypto_symbol is None:
        crypto_symbol = "BTC"
    if len(result_lst) == 0:
        return 0, []
    impacts = []

    for idx, post in enumerate(result_lst):
        text = get_text_from_post(post)
        labels = [crypto_symbol.lower(), CRYPTO_MAP[crypto_symbol.upper()]]
        labels.extend(GENERAL_CRYPTO_KEYWORDS)
        if keyword_filter(text, labels):
            # print(f"index: {idx}")
            scores = sia.polarity_scores(text)
            # print(f"{post['impact_score']}\n{scores}")
            impacts.append([idx, (post['impact_score'] * scores['compound'])])
            # print(f"impact: {post['impact_score'] * scores['compound']}\n")
    return sum(sublist[1] for sublist in impacts) / len(impacts), impacts


def get_sentiment_score_seq(posts: list, current_time: int = int(time.time()), seq_num: int = 5) -> list:
    """
    Call this Function to get Sequence of Sentiment Score in past 7*seq_num days

    :param posts: list of related posts
    :param current_time: time to start analyze, default now
    :param seq_num: number of element in the output list
    :return:
    """
    results = []
    for t in range(seq_num):
        start_time = current_time - t * MAX_NUMBERS['max_age']
        start_date = datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%d')
        end_date = datetime.utcfromtimestamp(start_time - MAX_NUMBERS['max_age']).strftime('%Y-%m-%d')
        logger.debug(f"Analyzing Post between {start_date} - {end_date} ...")
        x, lst = analyze_sentiment_now(posts, start_time)
        logger.debug(f"Number of Posts: {len(lst)}, Sentiment Score: {x}")
        logger.debug(f"Analyzing Finished ...")
        results.append(x)
    results.reverse()

    return results
