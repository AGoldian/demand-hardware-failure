# demand-hardware-failure

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
