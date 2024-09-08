import pandas as pd
from autogluon.tabular import TabularPredictor
import shutil
import os

# Функция для очистки папки перед сохранением новых моделей
def clear_weights_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Удаляем старую папку вместе с файлами
    os.makedirs(folder_path)  # Создаем пустую папку

# Класс AutoGluonModel, который содержит методы для обучения, предсказания и агрегации результатов
class AutoGluonModel:

    # Метод для обучения модели
    def fit(self, train_data: pd.DataFrame, save_path_model: str = 'weights_model', time_limit: int = 30, THRESH_NA: float = 0.5) -> pd.DataFrame:
        """
        Обучает модель с использованием AutoGluon.
        Аргументы:
        - train_data: данные для обучения в формате DataFrame
        - save_path_model: путь для сохранения модели (по умолчанию 'weights_model')
        - time_limit: ограничение по времени для обучения модели в секундах (по умолчанию 600)
        - THRESH_NA: порог для удаления столбцов с пропущенными значениями (по умолчанию 0.5)

        Возвращает таблицу с результатами лучшей модели (leaderboard).
        """

        # Удаляем столбец 'serial_number' и фильтруем колонки с пропущенными значениями выше THRESH_NA
        train_data = train_data.drop(columns=['serial_number'])
        train_data = train_data.loc[:, train_data.isnull().mean() < THRESH_NA]

        # Преобразуем sparse данные в плотный формат, если это необходимо
        for column in train_data.columns:
            if hasattr(train_data[column], "sparse") and train_data[column].sparse is not None:
                train_data[column] = train_data[column].sparse.to_dense()

        # Очищаем папку с моделями перед сохранением новой
        clear_weights_folder(save_path_model)

        # Обучаем модель с AutoGluon
        predictor = TabularPredictor(
            label='hard_live_cost',  # Целевая переменная
            problem_type='regression',  # Тип задачи - регрессия
            eval_metric='mean_absolute_error',  # Метрика оценки модели
            path=save_path_model,  # Путь для сохранения модели
        ).fit(
            train_data,
            time_limit=time_limit,  # Ограничение по времени
            presets='best_quality',  # Пресеты для качества модели
            keep_only_best=True  # Сохраняем только лучшую модель
        )

        # Возвращаем таблицу с результатами моделей (leaderboard)
        return predictor.leaderboard()

    # Метод для предсказания на локальной модели
    def predict_local_model(self, data: pd.DataFrame, save_path_model: str = 'weights_model') -> pd.DataFrame:
        """
        Загружает сохранённую модель и делает предсказания на новых данных.
        Аргументы:
        - data: входные данные в формате DataFrame

        Возвращает DataFrame с предсказанным количеством дней до выхода дисков из строя.
        """
        # Загружаем сохранённую модель
        loaded_predictor = TabularPredictor.load(save_path_model)

        # Сохраняем столбец 'serial_number', затем удаляем ненужные столбцы
        s_number = data['serial_number']
        data = data.drop(columns=['date', 'serial_number'])

        # Получаем предсказания
        predictions = loaded_predictor.predict(data)

        # Добавляем предсказания к данным
        data['predicted_days_to_failure'] = predictions
        data['serial_number'] = s_number
        predict_data = data[['serial_number', 'model', 'capacity_bytes', 'predicted_days_to_failure']]
        predict_data.to_csv('local_predict_model.csv', index=False)
        return predict_data

    # Метод для предсказания на глобальном уровне (агрегированные результаты)
    def predict_global_model(self, data_predict_local_model: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразует предсказания дней в месяцы и группирует результаты по временным интервалам.
        Аргументы:
        - data_predict_local_model: DataFrame с локальными предсказаниями

        Возвращает DataFrame с количеством дисков, которые выйдут из строя в течение следующих 3, 6, 9 и 12 месяцев.
        """

        # Преобразуем предсказанные дни до выхода из строя в месяцы
        data_predict_local_model['predicted_months_to_failure'] = data_predict_local_model['predicted_days_to_failure'] / 30

        # Определение временных интервалов
        def assign_time_interval(months):
            if months <= 3:
                return '0-3 месяца'
            elif months <= 6:
                return '4-6 месяцев'
            elif months <= 9:
                return '7-9 месяцев'
            elif months <= 12:
                return '10-12 месяцев'
            else:
                return 'Более 12 месяцев'

        # Применяем функцию для определения временных интервалов
        data_predict_local_model['time_interval'] = data_predict_local_model['predicted_months_to_failure'].apply(assign_time_interval)

        # Фильтруем только те диски, которые выйдут из строя в течение следующих 12 месяцев
        data_filtered = data_predict_local_model[data_predict_local_model['time_interval'] != 'Более 12 месяцев']

        # Группируем данные по ёмкости, модели и временному интервалу, подсчитывая количество дисков
        result = data_filtered.groupby(
            ['capacity_bytes', 'model', 'time_interval']
        ).size().reset_index(name='disk_count')

        # Упорядочиваем временные интервалы
        interval_order = ['0-3 месяца', '4-6 месяцев', '7-9 месяцев', '10-12 месяцев']
        result['time_interval'] = pd.Categorical(
            result['time_interval'],
            categories=interval_order,
            ordered=True
        )

        # Возвращаем отсортированные данные
        return pd.DataFrame(result, index=[i for i in range(len(result))]).sort_values(by=['time_interval', 'capacity_bytes', 'model'])

    # Метод для обучения и предсказания на тестовых данных
    def fit_predict(self, train_data: pd.DataFrame, test_data: pd.DataFrame, save_path_model: str = 'weights_model', time_limit: int = 600, THRESH_NA: float = 0.5):
        """
        Обучает модель, делает предсказания на тестовых данных и сохраняет результаты.
        Аргументы:
        - train_data: данные для обучения в формате DataFrame
        - test_data: данные для тестирования в формате DataFrame
        - save_path_model: путь для сохранения модели (по умолчанию 'weights_model')
        - time_limit: ограничение по времени для обучения модели в секундах (по умолчанию 600)
        - THRESH_NA: порог для удаления столбцов с пропущенными значениями (по умолчанию 0.5)

        Возвращает глобальные и локальные предсказания.
        """

        # Обучаем модель и получаем leaderboard
        leaderboard = self.fit(train_data, save_path_model, time_limit, THRESH_NA)
        print(leaderboard)

        # Делаем локальные предсказания
        local_predict_data = self.predict_local_model(test_data, save_path_model=save_path_model)
        local_predict_data.to_csv('local_predict_model.csv', index=False)  # Сохраняем локальные предсказания в CSV

        # Делаем глобальные предсказания
        global_predict = self.predict_global_model(local_predict_data)
        global_predict.to_csv('global_predict_model.csv', index=False)  # Сохраняем глобальные предсказания в CSV

        return global_predict, local_predict_data
