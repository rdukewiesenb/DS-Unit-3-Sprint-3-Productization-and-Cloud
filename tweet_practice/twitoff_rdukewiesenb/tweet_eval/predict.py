"""Predicting users based on embedded tweets"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .model import User
from .twitter import vectorize_tweet
# change model to classification

def predict_user(user0, user1, hypothetical_tweet):
    """
    Determine and return which user is more likely
    to say a given tweet.

    Example: predict_user(
        'jonathanvswan', 'alaynatreene', 
        'The president reported today that he is no
        longer sick from the virus'
    )
    return 0 (reporter1_name) or 1 (reporter2_name)
    """
    user0 = User.query.filter(User.name=='brittanys').one()
    user1 = User.query.filter(User.name=='jonathanvswan').one()
    user2 = User.query.filter(User.name=='kaitlancollins').one()
    user3 = User.query.filter(User.name=='Yamiche').one()
    user4 = User.query.filter(User.name=='anniekarni').one()
    user5 = User.query.filter(User.name=='weijia').one()
    user6 = User.query.filter(User.name=='AprilDRyan').one()
    user7 = User.query.filter(User.name=='jeffmason1').one()

    # Vectorize
    user0_vect = np.array([tweet.vect for tweet in user0.tweets])
    user1_vect = np.array([tweet.vect for tweet in user1.tweets])
    user2_vect = np.array([tweet.vect for tweet in user2.tweets])
    user3_vect = np.array([tweet.vect for tweet in user3.tweets])
    user4_vect = np.array([tweet.vect for tweet in user4.tweets])
    user5_vect = np.array([tweet.vect for tweet in user5.tweets])
    user6_vect = np.array([tweet.vect for tweet in user6.tweets])
    user7_vect = np.array([tweet.vect for tweet in user7.tweets])

    vects = np.vstack([user0_vect, user1_vect, user2_vect, user3_vect, user4_vect, user5_vect])
    # find alternative for this argument:
    labels = np.concatenate(
        [np.zeroes(len(user0.tweets)),
         np.ones(len(user1.tweets)),
         np.twoes(len(user2.tweets)),
         np.threes(len(user3.tweets)),
         np.fours(len(user4.tweets)),
         np.fives(len(user5.tweets)),
         np.sixes(len(user6.tweets)),
         np.sevens(len(user7.tweets))]
    )
    log_reg = LogisticRegression().fit(vects, labels)
    hypothetical_tweet = vectorize_tweet(hypothetical_tweet)
    return log_reg.predict(np.array(hypothetical_tweet).reshape(1, -1))


