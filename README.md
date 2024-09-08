# demand-hardware-failure

# Документация к CLI инструменту для обучения и предсказания моделей AutoGluon

Этот скрипт представляет собой CLI (Command Line Interface) инструмент для обучения и предсказания моделей с использованием библиотеки `AutoGluon`. Он позволяет пользователю выполнять три основные операции: обучение модели, предсказание результатов и комбинацию этих двух операций (обучение и предсказание).

## Установка зависимостей

Перед использованием скрипта убедитесь, что у вас установлены все необходимые зависимости. Вы можете установить их с помощью `pip`:

```bash
pip install -r requirements.txt
```

# Параметры и опции
- file_path: (обязательный аргумент) Путь к входному файлу, который будет использоваться для обучения или предсказания.
- second_file_path: (опциональный аргумент) Путь ко второму входному файлу, который используется только в случае вызова метода fit_predict.
- --fit: (опция) Флаг, указывающий на необходимость вызова метода обучения модели.
- --predict: (опция) Флаг, указывающий на необходимость вызова метода предсказания модели.
- --fit_predict: (опция) Флаг, указывающий на необходимость вызова метода обучения и предсказания модели.
- --preprocessing: (опция) Флаг, указывающий на необходимость обработки данных, подается папка с ежедневными наблюдениями.
- --help: (опция) Флаг, вызов функции помощи.

# Примеры использования
## Обучение модели

`python cli.py path/to/folder_data --preprocessing`
Этот пример запускает процесс обработки данных для модели, используя данные из папки path/to/folder_data, на выходе получаем computing_target_data.csv

## Обучение модели

`python cli.py path/to/train_data.csv --fit`
Этот пример запускает процесс обучения модели, используя данные из файла train_data.csv. На выходе ожидается путь к файлу с сохраненными весами.

## Предсказание результатов

`python cli.py path/to/test_data.csv --predict`
Этот пример запускает процесс предсказания результатов, используя данные из файла test_data.csv. Прежде запускается локальная модель, её результат сохраняется в папку и запускает глобальную модель, её результат выводться в консоль.

## Обучение и предсказание!

`python cli.py path/to/train_data.csv path/to/test_data.csv --fit_predict`
Этот пример запускает процесс обучения и предсказания, используя данные из файлов train_data.csv и test_data.csv. Обратите внимание, что для этой операции оба пути к файлам являются обязательными.

### Пример прогноза локальной модели
![img2](https://github.com/AGoldian/demand-hardware-failure/blob/production/src/local_model_output.png?raw=true)

### Пример прогноза глобальной модели
![img1](https://github.com/AGoldian/demand-hardware-failure/blob/production/src/global_model_output.png?raw=true)

## Внутреннее API

### Модуль stats

#### `iter_file(filepath: str) -> Generator[tuple[datetime, str, str, bool], None, None]`

Генерирует строки из CSV-файла

#### `iter_files(data_path: str) -> Generator[tuple[datetime, str, str, bool], None, None]`

Генерирует строки из папки с CSV-файлов

#### `get_data(folder: str = 'data') -> Generator[FailureInfo, None, None]`

Генерирует модели `FailureInfo()` по всем CSV-файлам в папке `folder` 

```python
@dataclass
class FailureInfo:
    serial_number: str
    model: str
    start_date: datetime
    failure_date: datetime
```

#### `init_sqlite3(conn)`

Инициализация SQLite3 базы данных и таблицы `failure_info`, содержащая четыре колонки:

- `serial_number` - серийный номер
- `model` - модель
- `start_date` - дата начала работы
- `failure_date` - дата отказа

#### `fill_data(conn, data_generator: Iterable[FailureInfo]) -> None`

Заполняет таблицу `failure_info` соответствующими сущностями

Используется как обучение или дообучение

#### `get_statistics_by_models(conn, percentile: float = 0.9) -> list[tuple[str, float]]`

Получает список перцентилей дней отказов по каждой модели, опираясь на таблицу `failure_info`

Используется как предсказание