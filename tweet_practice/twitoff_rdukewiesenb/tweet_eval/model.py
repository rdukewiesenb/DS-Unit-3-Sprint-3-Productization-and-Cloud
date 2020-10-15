from flask_sqlalchemy import SQLAlchemy


DB = SQLAlchemy()

class User(DB.Model):
    id = DB.Column(DB.Integer, primary_key=True)
    name = DB.Column(DB.String, nullable=False)
    newest_tweet_id = DB.Column(DB.BigInteger)

    def __repr__(self):
        return "<User: {}>".format(self.name)

class Tweet(DB.Model):
    id = DB.Column(DB.BigInteger, primary_key=True)
    text = DB.Column(DB.Unicode(300))
    vect = DB.Column(DB.PickleType, nullable=False)
    user_id = DB.Column(DB.BigInteger, DB.ForeignKey('user.id'), nullable=False)
    user = DB.relationship("User", backref=DB.backref("tweets", lazy=True))

    def __repr__(self):
        return "<Tweet: {}>".format(self.text)

# def example_users():
#     reporter0 = User(id=1, name="Jonathan Swan", user='jonathanvswan')
#     reporter1 = User(id=2, name="Alayna Treene", user='alaynatreene')
#     DB.session.add(reporter1)
#     DB.session.add(reporter2)
#     DB.session.commit()



    # each team must have an individual ID; `user` tells the computer which twitter account
    # the tweet was published through
   

"""
MISCALLANEOUS:
    t1 = Tweet(id=1, text="Whats excel and why am I highly proficient in it on all my job applications", user=brian)
    t2 = Tweet(id=3, text="compré mi pasaje pa PR pa sanguivin im so happy < 3", user=adriana)
    t3 = Tweet(id=4, text="What's the best 80s song of all time and why is it Power of Love by Huey Lewis and the News?", user=adriana)
    t4 = Tweet(id=5, text="AMIGUES MAMI Y PAPI ESTAN EN EL PIP CON DALMAU AHORA MISMO RECOGIENDO LETRERO PARA PONER EN CASA QUIERO RECONOCER EL GRAN ESFUERZO DE MI HERMANA @victoriaisabelr AYUDÁNDOME CONVENCERLOS A VOTAR POR UNA PATRIA NUEVA", user=adriana)
    t5 = Tweet(id=6, text="Should I post my resume on Twitter. I don’t think I’ll find a job, but I want some attention to hold me over ‘till I make dinner.", user=brian)
    t6 = Tweet(id=7, text="Proud to announce my book is ranked 3,360,948 on Amazon. I’d like to thank my alma mater and all the exes who ruined my life for making this possible", user=brian)
    DB.session.add(t1)
    DB.session.add(t2)
    DB.session.add(t3)
    DB.session.add(t4)
    DB.session.add(t5)
    DB.session.add(t6)
    DB.session.commit()
"""