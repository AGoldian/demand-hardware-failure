import click
from datetime import datetime

from model.train import AutoGluonModel

import pandas as pd


@click.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.argument('second_file_path', type=click.Path(exists=True), required=False)
@click.option('--fit', is_flag=True, help='Flag to call the fit method')
@click.option('--predict', is_flag=True, help='Flag to call the predict method')
@click.option('--fit_predict', is_flag=True, help='Flag to call the fit_predict method')
def main(file_path, second_file_path, fit, predict, fit_predict):
    model = AutoGluonModel()

    if fit_predict:
        if second_file_path is None:
            click.echo("Error: --fit_predict requires a second file path")
            return

        train_data = pd.read_csv(file_path)
        test_data = pd.read_csv(second_file_path)
        global_predict, local_predict_data = model.fit_predict(train_data, test_data)
        click.echo(global_predict)
        name_local_predict = f"{datetime.now()}-local_predict_model.csv"
        local_predict_data.to_csv(name_local_predict, index=False)
        click.echo(f"Local predict data in: {name_local_predict}")
        click.echo("Fit and predict completed. Results saved to 'local_predict_model.csv' and 'global_predict_model.csv'")

    elif fit:
        train_data = pd.read_csv(file_path)
        leaderboard = model.fit(train_data)
        click.echo("Fit completed. Leaderboard:")
        click.echo(leaderboard)

    elif predict:
        test_data = pd.read_csv(file_path)
        local_predict_data = model.predict_local_model(test_data)
        name_local_predict = f"{datetime.now()}-local_predict_model.csv"
        local_predict_data.to_csv(name_local_predict, index=False)
        click.echo(f"Local predict data in: {name_local_predict}")
        global_predict_data = model.predict_global_model(local_predict_data)
        click.echo("Predict completed. Results saved to 'local_predict_model.csv'")
        click.echo(global_predict_data)

    else:
        click.echo("No valid flag provided. Please use --fit, --predict, or --fit_predict.")


if __name__ == "__main__":
    main()