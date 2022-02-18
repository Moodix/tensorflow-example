import json

from PIL import Image
from werkzeug.wrappers import Request, Response
#import pandas as pd
#import joblib

from utils.image import predict_image, process_image
from utils.model import load_model

model = None


def predict(environ, start_response):
    # Load input image data from the HTTP request
    request = Request(environ)
    if not request.files:
        return Response('no file uploaded', 400)(environ, start_response)
    image_file = next(request.files.values())
    #testset = pd.read_csv(image_file)



    # The predictor must be lazily instantiated;
    # the TensorFlow graph can apparently not be shared
    # between processes.
    global model
    #if not model:
        #model = joblib.load('model.h5')
    #one_hot_encoded_data2 = pd.get_dummies(testset, columns = ['Code'])
    #df2 = one_hot_encoded_data2[["Delay", "Code_200", "Code_201", "Code_204", "Code_302", "Code_400", "Code_404", "Code_500","Y"]]
    #X_test=df2[["Delay", "Code_200", "Code_201", "Code_204", "Code_302", "Code_400", "Code_404", "Code_500"]]
    #y_test=df2[["Y"]]
    #prediction = model.predict(X_test)

    # The following line allows Valohai to track endpoint predictions
    # while the model is deployed. Here we remove the full predictions
    # details as we are only interested in tracking the rest of the results.
    print(json.dumps({'vh_metadata': {k: v for k, v in prediction.items() if k != 'predictions'}}))

    # Return a JSON response
    #response = Response(json.dumps(prediction), content_type='application/json')
    d = {"foo": "bar"}
    response = Response(json.dumps(d), content_type='application/json')
    return response(environ, start_response)


# Run a local server for testing with `python deploy.py`
if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', 8000, predict)
