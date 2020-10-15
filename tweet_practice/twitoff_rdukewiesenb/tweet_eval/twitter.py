"""Retrieve tweets, word embeddings and populate DB"""

import tweepy
import spacy
from .model import DB, Tweet, User
from os import getenv

TWITTER_API_KEY = getenv('TWITTER_API_KEY')
TWITTER_API_KEY_SECRET = getenv('TWITTER_API_KEY_SECRET')
TWITTER_ACCESS_TOKEN = getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = getenv('TWITTER_ACCESS_TOKEN_SECRET')
TWITTER_AUTH=tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_KEY_SECRET)
TWITTER_AUTH.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
TWITTER=tweepy.API(TWITTER_AUTH)

# """This function will return a numpy.array"""

nlp = spacy.load('my_model')
def vectorize_tweet(tweet_text):
    return nlp(tweet_text).vector

def add_or_update_user(username):
    # """Allows us to add or update users to our DB"""
    try: 
        twitter_user = TWITTER.get_user(username)
        db_user = (User.query.get(twitter_user.id)) or User(id=twitter_user.id, name=username)
        DB.session.add(db_user)

        tweets = twitter_user.timeline(
            count=200, exclude_replies=True, 
            include_rts=False, tweet_mode='extended',
            since_id=db_user.newest_tweet_id
        )

        if tweets:
            db_user.newest_tweet_id = tweets[0].id

    # """This for loop is built assuming that all tweets are new"""
        for tweet in tweets:
            vectorized_tweet = vectorize_tweet(tweet.full_text)
            db_tweet = Tweet(
                id=tweet.id, text=tweet.full_text,
                vect=embedded_tweet
            )
            db_user.tweets.append(db_tweet)
            DB.session.add(db_tweet)

    # """This except statement is to catch any accounts that don't exist"""
    except Exception as e:
        print('Error Processing: {}: {}:'.format(username, e))
        raise e

    else:
        DB.session.commit()

def update_all_users():
    for user in User.query.all():
        add_or_update_user(user.name)


# def example_users():
#     add_or_update_user('alaynatreene')
#     add_or_update_user('jonathanvswan')

    # add_or_update_user('weijia')
    # add_or_update_user('kaitlancollins')
    # add_or_update_user('Acosta')
    # add_or_update_user('ZekeJMiller')
    # add_or_update_user('maggieNYT')
    # add_or_update_user('Yamiche')

