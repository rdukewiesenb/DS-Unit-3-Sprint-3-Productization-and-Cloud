from flask import Flask, render_template, request
from .twitter import example_users, add_or_update_user
from .model import DB, User, Tweet
from .predict import predict_user

def create_app():
    app = Flask(__name__)
    app.config['SQLAlchemy_DATABASE_URI'] = "sqlite3://db.sqlite3"
    app.config['SQLQLCHEMY_TRACK_MODIFICATIONS'] = False
    DB.init_app(app)

    @app.route('/')
    def root():
        return render_template("base.html", title="Home", users=User.query.all())

    @app.route('/compare', methods=['POST'])
    def compare():
        reporter0, reporter1 = sorted(
            [request.values['user1'],
            request.values['user2']]
        )
        if reporter0 == reporter1:
            message = "Cannot compare users to themselves!"
        else:
            prediction = predict_user(reporter0, reporter1, request.values['tweet_text'])
            message = "{} is more likely to be said by {} than {}".format(
                request.values['tweet_text'], reporter1 if prediction else reporter0,
                reporter0 if prediction else reporter1)
        
        return render_template('prediction.html', title='Prediction', message=message)

    @app.route('/user', methods=['POST'])
    @app.route('/user/<name>', methods=['GET'])
    def user(name=None, message=''):
        name = name or request.values['user_name']
        try:
            if request.method == 'POST':
                add_or_update_user(name)
                message = 'User {} successfully added!'.format(name)

            tweets = User.query.filter(User.name == name).one().tweets

        except Exception as e:
            message = "Error adding {}: {}".format(name, e)
            tweets = []
        
        return render_template('user.html', title=name, tweets=tweets, message=message)

    @app.route('/update')
    def update():
        # adds our users
        insert_example_users()
        return render_template('base.html', title="Home", users=User.query.all())

    @app.route('/reset')
    def reset():
        # resets database
        DB.drop_all()
        DB.create_all()
        return render_template('base.html', title='Reset Database!')

    return app


"""
IN TERMINAL: 
    IN FLASK SHELL:
        from tweet_eval.model import *
        example_users()
        User.query.all()



        from tweet_eval.model import User, Tweet
        user = User(id=1, name='user')
        tweet = Tweet(id=1, text='test', user=user)
        user.tweets

       test = User.query.first()
       test.tweets
"""