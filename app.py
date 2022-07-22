import flask
import pickle
import pandas as pd
import flask_debugtoolbar
from flask_debugtoolbar import DebugToolbarExtension
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

#initalize flask
app = flask.Flask(__name__)

# debug console, set to Ture for debug
app.debug = False
app.config['SECRET_KEY'] = 'password'
toolbar = DebugToolbarExtension(app)

# traffic protection
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["2 per minute", "1 per second"],
)

#load prediction model
pipe = pickle.load(open("pipe.pkl", 'rb'))

#initalize homepage
@app.route('/')
@limiter.limit("20 per minute")
def index():
    return flask.render_template('index.html')

#initalize results page
@app.route('/age', methods=['POST'])
@limiter.limit("6 per minute")
def predict_age():
    args = flask.request.form
    print(args)
    data = pd.DataFrame({
        'shell_weight': [float(args.get('shell_weight'))],
        'viscera_weight': [float(args.get('viscera_weight'))],
        'shucked_weight': [float(args.get('shucked_weight'))],
        'whole_weight': [float(args.get('whole_weight'))],
        'height': [float(args.get('height'))],
        'length': [float(args.get('length'))],
        'diameter': [float(args.get('diameter'))],
        'sex': [args.get('sex')]
    })
    prediction = str(round(pipe.predict(data)[0], 1))
    return flask.render_template('result.html', prediction=prediction)

#run the webapp
if __name__ == '__main__':
    app.run(port=5001, debug=True)
