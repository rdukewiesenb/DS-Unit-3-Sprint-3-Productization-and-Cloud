"""
STEPS IN TERMINAL:

Importing tweepy into Flask shell:
    import tweepy
    TWITTER_API_KEY='[Insert API Key]'
    TWITTER_API_KEY_SECRET='[Insert Secret API Key]'
    auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_KEY_SECRET)
    twitter = tweepy.API(auth)
    user = '[twitter handle]'
    twitter_user = twitter.get_user(user)
    twitter_user.id
    twitter_user.create_at


Importing .twitter into Flask shell:
    from tweet_eval.model import User, DB, Tweet
    from tweet_eval.twitter import add_or_update_user
    add_or_update_user('ziwe')

    What happens when we try to analyze an account,
    but find out that it doesn't exist?
        add_or_update_user('fakeaccount')


SQL Query in Flask shell:
    user1 = User.query.filter(User.name == 'ziwe')
    len(user1.tweets)

Importing spacy into repl:
    import spacy
    import en_core_web_sm
    nlp = en_core_web_sm.load()
    nlp
    nlp.to_disk('my_model/')
    nlp = spacy.load('my_model')
    word2vect = nlp('testing out nlp').vector
    word2vect
     word2vect_2 = nlp('We are doing another text').vector     
    len(word2vect_2)
"""

"""
(twitoff_rdukewiesenb) rdukewiesenb@LAPTOP-F6MJ9381:/mnt/c/Users/rduke/Repositories/DS-Unit-3-Sprint-3-Productization-and-Cloud/tweet_practice/twitoff_rdukewiesenb$ flask shell
/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/flask_sqlalchemy/__init__.py:812: UserWarning: Neither SQLALCHEMY_DATABASE_URI nor SQLALCHEMY_BINDS is set. Defaulting SQLALCHEMY_DATABASE_URI to 
"sqlite:///:memory:".
  warnings.warn(
/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/flask_sqlalchemy/__init__.py:833: FSADeprecationWarning: SQLALCHEMY_TRACK_MODIFICATIONS adds significant overhead and will be disabled by default 
in the future.  Set it to True or False to suppress this warning.
  warnings.warn(FSADeprecationWarning(
Python 3.8.2 (default, Jul 16 2020, 14:00:26) 
[GCC 9.3.0] on linux
App: tweet_eval.app [production]
Instance: /mnt/c/Users/rduke/Repositories/DS-Unit-3-Sprint-3-Productization-and-Cloud/tweet_practice/twitoff_rdukewiesenb/instance
>>> from tweet_eval.model import User
>>> from tweet_eval.model import User, Tweet, DB
>>> from tweet_eval.twitter import add_or_update_user
>>> DB.drop_all()
>>> DB.create_all()
>>> User.query.all()
[]
>>> add_or_update_user('jonathanvswan')
Error Processing: jonathanvswan: name 'Flase' is not defined:
Traceback (most recent call last):
  File "<console>", line 1, in <module>
  File "/mnt/c/Users/rduke/Repositories/DS-Unit-3-Sprint-3-Productization-and-Cloud/tweet_practice/twitoff_rdukewiesenb/tweet_eval/twitter.py", line 50, in add_or_update_user
    raise e
  File "/mnt/c/Users/rduke/Repositories/DS-Unit-3-Sprint-3-Productization-and-Cloud/tweet_practice/twitoff_rdukewiesenb/tweet_eval/twitter.py", line 31, in add_or_update_user
    include_rts=Flase, tweet_mode='extended'
NameError: name 'Flase' is not defined
>>> exit()
(twitoff_rdukewiesenb) rdukewiesenb@LAPTOP-F6MJ9381:/mnt/c/Users/rduke/Repositories/DS-Unit-3-Sprint-3-Productization-and-Cloud/tweet_practice/twitoff_rdukewiesenb$ flask shell
/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/flask_sqlalchemy/__init__.py:812: UserWarning: Neither SQLALCHEMY_DATABASE_URI nor SQLALCHEMY_BINDS is set. Defaulting SQLALCHEMY_DATABASE_URI to 
"sqlite:///:memory:".
  warnings.warn(
/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/flask_sqlalchemy/__init__.py:833: FSADeprecationWarning: SQLALCHEMY_TRACK_MODIFICATIONS adds significant overhead and will be disabled by default 
in the future.  Set it to True or False to suppress this warning.
  warnings.warn(FSADeprecationWarning(
Python 3.8.2 (default, Jul 16 2020, 14:00:26) 
[GCC 9.3.0] on linux
App: tweet_eval.app [production]
Instance: /mnt/c/Users/rduke/Repositories/DS-Unit-3-Sprint-3-Productization-and-Cloud/tweet_practice/twitoff_rdukewiesenb/instance
>>> from tweet_eval.model import User, DB, Tweet
>>> from tweet_eval.twitter import add_or_update_user
>>> User.query.all()
Traceback (most recent call last):
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sqlalchemy/engine/base.py", line 1276, in _execute_context
    self.dialect.do_execute(
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sqlalchemy/engine/default.py", line 593, in do_execute
    cursor.execute(statement, parameters)
sqlite3.OperationalError: no such table: user

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<console>", line 1, in <module>
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sqlalchemy/orm/query.py", line 3373, in all
    return list(self)
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sqlalchemy/orm/query.py", line 3535, in __iter__
    return self._execute_and_instances(context)
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sqlalchemy/orm/query.py", line 3560, in _execute_and_instances
    result = conn.execute(querycontext.statement, self._params)
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sqlalchemy/engine/base.py", line 1011, in execute
    return meth(self, multiparams, params)
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sqlalchemy/sql/elements.py", line 298, in _execute_on_connection
    return connection._execute_clauseelement(self, multiparams, params)
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sqlalchemy/engine/base.py", line 1124, in _execute_clauseelement
    ret = self._execute_context(
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sqlalchemy/engine/base.py", line 1316, in _execute_context
    self._handle_dbapi_exception(
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sqlalchemy/engine/base.py", line 1510, in _handle_dbapi_exception
    util.raise_(
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sqlalchemy/util/compat.py", line 182, in raise_
    raise exception
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sqlalchemy/engine/base.py", line 1276, in _execute_context
    self.dialect.do_execute(
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sqlalchemy/engine/default.py", line 593, in do_execute
    cursor.execute(statement, parameters)
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) no such table: user
[SQL: SELECT user.id AS user_id, user.name AS user_name, user.newest_tweet_id AS user_newest_tweet_id
FROM user]
(Background on this error at: http://sqlalche.me/e/13/e3q8)   
>>> DB.drop_all()
>>> DB.create_all()
>>> add_or_update_user('alaynatreene')
>>> add_or_update_user('jonathanvswan')
>>> reporter1 = User.query.filter(User.name=='alaynatreene').one()
>>> reporter1
<User: alaynatreene>
>>> reporter2 = User.query.filter(User.name=='jonathanvswan').one()
>>> reporter2
<User: jonathanvswan>
>>> reporter1.tweets[0]
<Tweet: .@SenJohnKennedy brought ACB's children blank pads and pencils to use as the hearing continues, per pooler @ESCochrane>
>>> from sklearn.linear_model import LogisticRegression
>>> LogisticRegression
<class 'sklearn.linear_model._logistic.LogisticRegression'>
>>> import numpy as np
>>> np
<module 'numpy' from '/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/numpy/__init__.py'>
>>> reporter2.tweets[0]
<Tweet: Trump airing internal polls, version 1,000. https://t.co/ej7rMonfdG>
>>> len(reporter1.tweets)
83
>>> len(reporter2.tweets)
65
>>> reporter1_vects = np.array([tweet.vect for tweet in reporter1.tweets])
>>> reporter1_vects.shape
(83, 96)
>>> reporter2_vects = np.array([tweet.vect for tweet in reporter2.tweets])
>>> reporter2_vects.shape
(65, 96)
>>> vects = np.vstack([reporter1_vects, reporter2_vects]) 
>>> vects.shape
(148, 96)
>>> reporter1_vects.shape[0] + repoter2_vects.shape[0]
Traceback (most recent call last):
  File "<console>", line 1, in <module>
NameError: name 'repoter2_vects' is not defined
>>> reporter1_vects.shape[0] + reporter2_vects.shape[0]
148
>>> labels = np.concatenate([np.zeros(len(reporter1.tweets), np.ones(reporter2.tweets))])
Traceback (most recent call last):
  File "<console>", line 1, in <module>
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/numpy/core/numeric.py", line 192, in ones
    a = empty(shape, dtype, order)
ValueError: maximum supported dimension for an ndarray is 32, 
found 65
>>> labels = np.concatenate([np.zeros(len(reporter1.tweets)), 
np.ones(len(reporter2.tweets))])
>>> log_reg = LogisticRegression().fit(vects, labels)
>>> log_reg
LogisticRegression()
>>> log_reg.predict
<bound method LinearClassifierMixin.predict of LogisticRegression()>
>>> log_reg.predict('Today at the White House')
Traceback (most recent call last):
  File "<console>", line 1, in <module>
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sklearn/linear_model/_base.py", line 307, in predict
    scores = self.decision_function(X)
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sklearn/linear_model/_base.py", line 282, in decision_function
    X = check_array(X, accept_sparse='csr')
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sklearn/utils/validation.py", line 72, in inner_f
    return f(**kwargs)
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sklearn/utils/validation.py", line 612, in check_array
    raise ValueError(
ValueError: Expected 2D array, got scalar array instead:      
array=Today at the White House.
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
>>> hypothetical = 'Today at the White House'
>>> from tweet_eval.twitter import vectorize_tweet
>>> import spacy
>>> hypothetical_vect = vectorize_tweet(hypothetical)
>>> log_reg.predict(hypothetical_vect)
Traceback (most recent call last):
  File "<console>", line 1, in <module>
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sklearn/linear_model/_base.py", line 307, in predict
    scores = self.decision_function(X)
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sklearn/linear_model/_base.py", line 282, in decision_function
    X = check_array(X, accept_sparse='csr')
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sklearn/utils/validation.py", line 72, in inner_f
    return f(**kwargs)
  File "/home/rdukewiesenb/.local/share/virtualenvs/twitoff_rdukewiesenb-kCxIt6d9/lib/python3.8/site-packages/sklearn/utils/validation.py", line 619, in check_array
    raise ValueError(
ValueError: Expected 2D array, got 1D array instead:
array=[-0.48999152 -1.0506601   0.04691834 -0.46974096  1.1842439  -1.0139414
  0.4352087   0.4102808  -1.1871628   2.2355835   0.86747265 -1.4082257
  0.9590987   0.8858334  -1.6738861  -0.16657045  0.20800011  
1.2468283
 -1.4744714   0.8000059  -0.01073804  1.6612341   0.9510249  -1.8388498
  0.28151625  1.1259946  -2.1590524  -0.9221485   0.07129027 -0.5771143
 -0.3775765  -1.142717   -0.3572192  -1.1650842  -1.5582409   
1.3367002
 -0.1147816  -0.84357184  0.21096471  0.14039361  2.6751077  -0.9059657
  0.19702509  0.76116204  1.0472091  -1.8295336   0.6023796   
1.8269014
  0.14258942  0.4486061  -1.5919753  -0.90692365 -0.03391633 -1.265733
 -1.8968999   2.420119    1.0299232   0.22965236 -1.1957599   
0.13186069
 -1.1149797   0.01447918  1.765966   -1.7544582  -0.20239635  
1.9018357
  0.0360441  -0.6314623   1.1217401   0.4674446  -1.2261994   
1.371577
  0.17502852 -1.1328067   0.9490164   0.33947402  0.03958728  
0.21303988
  0.46786016 -0.5477688   0.8429661   0.8301345  -1.4017885  -1.0654998
  1.0881249   1.3685142   1.0661527   1.106189    0.57071704  
0.33755943
 -1.3312762  -0.48989925 -1.5481374   0.40862948  1.00608    -0.41487485].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
>>> hypothetical_vect = np.array(hypothetical_vect).reshape(1, -1)
>>> hypothetical_vect.shape
(1, 96)
>>> log_reg.predict(hypothetical_vect)
array([0.])
>>> exit()
"""








