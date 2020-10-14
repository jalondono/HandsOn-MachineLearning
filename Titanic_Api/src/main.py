import uvicorn
from fastapi import FastAPI, Request
import pickle
import pandas as pd


def load_model():
    modelfile = 'src/model.pkl'
    # open a file, where you stored the pickled data
    file = open(modelfile, 'rb')

    model = pickle.load(file)
    file.close()
    print(model)
    return model

    # close the file


def predict(model, df):
    """
    make the predictions over the dataset
    :param model:
    :return:
    """
    y_pred = model.predict(df)
    data_pred = df.copy()
    data_pred['Survived'] = y_pred
    print(data_pred[['Name', 'Survived']].head(20))
    data_json = pd.DataFrame.to_dict(data_pred[['Name', 'Survived']])

    total = data_pred.shape[0]
    survived = sum(data_pred['Survived'] == 1)
    percentage = round(100 * survived / total, 2)

    print(f"Survived: {survived}/{total} or {percentage}%")
    return {"predictions": {"Survivors": data_json,
                            "Number_of_survivors": survived,
                            "Percentage": percentage,
                            "Number_of_people": total}}


app = FastAPI()
# model = load_model()


# Api root or home endpoint
@app.get('/')
@app.get('/home')
def read_home():
    """
    Home endpoint which can be used to test the availability
    of the application.
    :return: Dict with key 'message' and value 'API live!'
    """
    return {'message': 'API live!'}


@app.post("/api")
async def api_model(request: Request):
    model = load_model()
    data = await request.json()
    data = data["data"]
    df = pd.DataFrame(data)
    predictions = predict(model, df)
    return predictions


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
