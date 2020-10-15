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
    user0 = User.query.filter(User.name=='jonathanvswan').one()
    user1 = User.query.filter(User.name=='alaynatreene').one()
    user0_vect = np.array([tweet.vect for tweet in reporter0.tweets])
    user1_vect = np.array([tweet.vect for tweet in reporter1.tweets])

    vects = np.vstack([reporter0_vect, reporter1_vect])
    # find alternative for this argument:
    labels = np.concatenate(
        [np.zeroes(len(reporter0.tweets)),
         np.ones(len(reporter1.tweets))]
    )
    log_reg = LogisticRegression().fit(vects, labels)
    hypothetical_tweet = vectorize_tweet(hypothetical_tweet)
    return log_reg.predict(np.array(hypothetical_tweet).reshape(1, -1))


