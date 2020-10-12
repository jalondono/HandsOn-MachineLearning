"""loads a serialized model to make a prediction
"""
import typer
import pickle

load = __import__('utils').load_data
app = typer.Typer()


@app.command()
def predict(data_path: str = 'data/titanic.zip',
            name_file: str = 'test.csv',
            modelfile: str = '/home/jalondono/Holberton/HandsOn-MachineLearning/Titanic_deploy/mlruns/0'
                             '/c87244e5ca5f44848a6e99e68e80a37d/artifacts/model/model.pkl'):
    data = load(data_path, name_file, None)

    # open a file, where you stored the pickled data
    file = open(modelfile, 'rb')

    # load the model
    model = pickle.load(file)

    # close the file
    file.close()

    # make the prediction
    y_pred = model.predict(data)
    data_pred = data.copy()
    data_pred['Survived'] = y_pred
    print(data_pred[['Name', 'Survived']].head(20))

    total = data_pred.shape[0]
    survived = sum(data_pred['Survived'] == 1)
    percentage = round(100 * survived / total, 2)

    print(f"Survived: {survived}/{total} or {percentage}%")


if __name__ == "__main__":
    app()
