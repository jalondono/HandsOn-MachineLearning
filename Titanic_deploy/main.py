import uvicorn
from fastapi import FastAPI, Request
import pickle
import pandas as pd


def load_model():
    data_path = 'data/titanic.zip',
    name_file = 'test.csv',
    modelfile = '/home/jalondono/Holberton/HandsOn-MachineLearning/Titanic_deploy/mlruns/0' \
                '/c87244e5ca5f44848a6e99e68e80a37d/artifacts/model/model.pkl'
    # open a file, where you stored the pickled data
    file = open(modelfile, 'rb')

    # load the model
    model = pickle.load(file)

    # close the file
    file.close()
    return model


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
model = load_model()


@app.post("/")
async def api_model(request: Request):
    data = await request.json()
    data = data["data"]
    df = pd.DataFrame(data)
    predictions = predict(model, df)
    return predictions


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
