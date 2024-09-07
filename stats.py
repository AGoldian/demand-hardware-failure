import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Generator, Iterable


@dataclass
class FailureInfo:
    serial_number: str
    model: str
    start_date: datetime
    failure_date: datetime


def iter_file(filepath: str) -> Generator[tuple[datetime, str, str, bool], None, None]:
    with open(filepath) as file:
        date = datetime.strptime(filepath.split('/')[1].split('.csv')[0], '%Y-%m-%d')
        for line in file:
            items = line.split(',')
            yield date, items[1], items[2], items[4] == '1'


def iter_files(data_path: str) -> Generator[tuple[datetime, str, str, bool], None, None]:
    for filepath in sorted(os.listdir(data_path)):
        yield from iter_file(os.path.join(data_path, filepath))


def get_data(folder: str = 'data') -> Generator[FailureInfo, None, None]:
    first_log = {}
    for date, serial, model, failure in iter_files(folder):
        key = serial
        if not failure and key not in first_log:
            first_log[key] = date
        elif failure and key in first_log:
            yield FailureInfo(
                serial_number=serial,
                model=model,
                start_date=first_log[key],
                failure_date=date,
            )
            del first_log[key]


DATABASE_NAME = 'database.sqlite'


def init_sqlite3(conn) -> None:
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS failure_info (
        serial_number TEXT,
        model TEXT,
        start_date TEXT,
        failure_date TEXT
    )
    ''')


def fill_data(conn, data_generator: Iterable[FailureInfo]) -> None:
    cursor = conn.cursor()
    for failure_info in data_generator:
        cursor.execute('''
        INSERT INTO failure_info (serial_number, model, start_date, failure_date)
        VALUES (?, ?, ?, ?)
        ''', (
            failure_info.serial_number,
            failure_info.model,
            failure_info.start_date,
            failure_info.failure_date
        ))


def get_statistics_by_models(
    conn,
    percentile: float = 0.9,
) -> list[tuple[str, float]]:
    cursor = conn.cursor()
    cursor.execute('''
    WITH data AS (
        SELECT
            model,
            julianday(failure_date) - julianday(start_date) as diff
        FROM failure_info
    ),
        ranked AS (
            SELECT model,
                   diff,
                   row_number() over (ORDER BY diff) AS row_num,
                   COUNT(*) OVER () AS total_rows
            FROM data
        )
    SELECT model, AVG(diff) AS percentile
    FROM ranked
    WHERE row_num <= total_rows * ?
    GROUP BY model;
    ''', (percentile,))
    return cursor.fetchall()


if __name__ == '__main__':
    with sqlite3.connect(DATABASE_NAME) as conn:
        init_sqlite3(conn)
        fill_data(conn, get_data())
        print(*get_statistics_by_models(conn, 0.9), sep='\n')
