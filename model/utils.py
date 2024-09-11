import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def update_smart_features(new_df, output_df):
    """
    Обновляет и добавляет SMART признаки в выходной DataFrame.

    Параметры:
    new_df (pd.DataFrame): Новый DataFrame с обновленными SMART признаками.
    output_df (pd.DataFrame): Существующий DataFrame, который нужно обновить.

    Возвращает:
    pd.DataFrame: Обновленный DataFrame с новыми и обновленными SMART признаками.
    """
    # Префиксы для SMART атрибутов
    smart_prefixes = ['smart_{}_normalized'.format(i) for i in range(1, 256)] + \
                     ['smart_{}_raw'.format(i) for i in range(1, 256)]

    # Убедимся, что у нас индексы актуальны для обоих DataFrame
    if 'serial_number' in new_df.columns:
        new_df.set_index('serial_number', inplace=True)

    # Проверим, что все нужные колонки из smart_prefixes присутствуют в new_df
    missing_cols_in_new_df = [col for col in smart_prefixes if col not in new_df.columns]

    # Определим недостающие колонки, которых нет в output_df
    missing_features = [col for col in smart_prefixes if col not in output_df.columns]

    # Если есть пропущенные колонки в output_df, создаем их с NaN
    if missing_features:
        missing_data = pd.DataFrame(np.nan, index=output_df.index, columns=missing_features)
        output_df = pd.concat([output_df, missing_data], axis=1)

    # Определяем пересекающиеся индексы (serial numbers)
    common_serials = output_df.index.intersection(new_df.index)

    # Обновляем данные только для пересекающихся индексов и колонок, которые существуют в обоих DataFrame
    matching_columns = [col for col in smart_prefixes if col in new_df.columns and col in output_df.columns]

    if len(common_serials) > 0 and len(matching_columns) > 0:
        # Выполняем обновление для пересечений
        output_df.loc[common_serials, matching_columns] = new_df.loc[common_serials, matching_columns]

    # Добавим новые строки из new_df, которых еще нет в output_df
    missing_serials = new_df.index.difference(output_df.index)

    if not missing_serials.empty:
        # Берем строки для отсутствующих serial_number
        new_data_to_add = new_df.loc[missing_serials, smart_prefixes]
        output_df = pd.concat([output_df, new_data_to_add])

    return output_df


def load_existing_output(output_path):
    """
    Загружает существующий DataFrame из указанного CSV файла или создает пустой DataFrame.

    Параметры:
    output_path (str): Путь к CSV файлу для загрузки.

    Возвращает:
    pd.DataFrame: Существующий DataFrame или новый пустой DataFrame с нужными колонками.
    """
    if output_path is not None and os.path.exists(output_path):
        # Загружаем уже существующие данные из CSV
        existing_data = pd.read_csv(output_path)

        # Устанавливаем serial_number как индекс, если столбец существует
        if 'serial_number' in existing_data.columns:
            existing_data.set_index('serial_number', inplace=True)

    else:
        # Если файла нет, создаем пустой DataFrame с нужными колонками

        # Стандартные колонки
        base_columns = ['model', 'capacity_bytes', 'hard_live_cost']

        # SMART признаки
        smart_columns = ['smart_{}_normalized'.format(i) for i in range(1, 256)] + \
                        ['smart_{}_raw'.format(i) for i in range(1, 256)]

        # Полный список колонок
        columns = base_columns + smart_columns

        # Создаем пустой DataFrame с колонками
        existing_data = pd.DataFrame(columns=columns)
        existing_data.index.name = 'serial_number'  # Устанавливаем serial_number как индекс для единообразия

    return existing_data


def update_features(new_df, output_new_df, is_last_file=False):
    """
    Обновляет признаки и добавляет новые диски в выходной DataFrame.

    Параметры:
    new_df (pd.DataFrame): Новый DataFrame с обновленными данными о дисках.
    output_new_df (pd.DataFrame): Существующий DataFrame, который нужно обновить.
    is_last_file (bool): Флаг, указывающий, является ли текущий файл последним в обработке.

    Возвращает:
    pd.DataFrame: Обновленный DataFrame с новыми и обновленными данными.
    """
    # Разделим новые строки на существующие и новые диски
    existing_disks = new_df[new_df['serial_number'].isin(output_new_df.index)]  # Существующие ряды
    new_disks = new_df[~new_df['serial_number'].isin(output_new_df.index)]  # Новые ряды

    # Обновим "hard_live_cost" для существующих дисков, где `failure == 0`
    if not existing_disks.empty:
        # Увеличиваем `hard_live_cost` на 1 только для строк, где failure==0
        mask = existing_disks['failure'] == 0
        update_serials = existing_disks[mask]['serial_number']

        # Обновляем hard_live_cost только для записей, где failure == 0
        if not is_last_file:
            output_new_df.loc[update_serials, 'hard_live_cost'] += 1
        else:
            output_new_df.loc[update_serials, 'hard_live_cost'] += 2000

    # Добавим новые строки для новых дисков
    if not new_disks.empty and not is_last_file:
        # Подготавливаем DataFrame для новых строк
        new_disks_to_add = new_disks.set_index('serial_number')[
            ['model', 'capacity_bytes']]  # устанавливаем нужные колонки для добавления
        new_disks_to_add['hard_live_cost'] = 1  # hard_live_cost инициализируем как 1

        # Конкатенация новых строк
        output_new_df = pd.concat([output_new_df, new_disks_to_add])

    return output_new_df


def compute_targets(folder_path: str, feature_file_path=None):
    """
    Основная функция для обработки файлов и обновления целевых данных.

    Параметры:
    folder_path (str): Путь к папке с CSV файлами.
    feature_file_path (str): Путь к CSV файлу с существующими данными (опционально).

    Возвращает:
    output_new_df (pd.DataFrame): набор serial_number дисков с признаками и таргетом - сколько прожили.
    """
    # Загружаем существующий выходной DataFrame
    output_new_df = load_existing_output(feature_file_path)

    # Получаем список всех файлов в папке
    all_files = os.listdir(folder_path)
    # Фильтруем только .csv файлы
    csv_files = [file for file in all_files if file.endswith('.csv')]

    checkpoint_dir = 'checkpoints_feature_compute'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Обрабатываем каждый файл с помощью прогресс-бара
    for idx, csv_data in enumerate(tqdm(csv_files, desc="Обработка файлов")):
        is_last_file = (idx == len(csv_files) - 1)

        if csv_data.endswith('.csv'):
            # Считываем новый DataFrame
            new_df = pd.read_csv(os.path.join(folder_path, csv_data))
            # Обновляем данные
            try:
                output_new_df = update_features(new_df, output_new_df, is_last_file=is_last_file)
                output_new_df = update_smart_features(new_df, output_new_df)
            except:
                pass
            print(csv_data, 'Обработан')

        # Сохраняем промежуточный результат каждые 90 файлов
        if idx % 90 == 0:
            output_new_df.to_csv(f'{checkpoint_dir}/computing_target_data{idx}.csv')

    # Сохраняем финальный результат
    output_new_df.to_csv('computing_target_data.csv')

    return output_new_df
