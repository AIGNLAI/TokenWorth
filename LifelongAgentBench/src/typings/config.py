from pydantic import BaseModel, Field, field_validator
from typing import Literal, Sequence, Mapping
import datetime
import re

from .general import SampleIndex
from .instance_factory import GeneralInstanceFactory


SampleOrderDescription = Literal["default"]  # Execute all the samples one by one


class LoggerConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    # See ConfigUtility.read_raw_config for the default value of log_file_path
    log_file_path: str
    logger_name: str


class AssignmentConfig(BaseModel):
    task: GeneralInstanceFactory
    agent: GeneralInstanceFactory
    language_model_dict: Mapping[str, GeneralInstanceFactory]
    callback_dict: Mapping[str, GeneralInstanceFactory]
    output_dir: str
    sample_order: Sequence[SampleIndex] | SampleOrderDescription
    generation_budget: int | None = Field(default=None, gt=0)

    @field_validator("output_dir", mode="before")  # noqa
    @classmethod
    def output_path_validation(cls, value: str) -> str:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        predefined_structure = {
            "TIMESTAMP": timestamp,
            "TIMESTAMP_MS": f"{timestamp}-{now.microsecond // 1000:03d}",
            "TIMESTAMP_DATE": now.strftime("%Y-%m-%d"),
            "TIMESTAMP_TIME": now.strftime("%H-%M-%S"),
        }
        assert isinstance(
            value, str
        ), f"'output_dir' must be a string, but got {type(value)}"
        return value.format(**predefined_structure)

    @staticmethod
    def is_output_dir_equal(output_dir_1: str, output_dir_2: str) -> bool:
        def recover_output_dir(_output_dir: str) -> str:
            timestamp_ms_pattern = (
                r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3}"
            )
            timestamp_pattern = r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
            date_pattern = r"\d{4}-\d{2}-\d{2}"
            time_pattern = r"\d{2}-\d{2}-\d{2}"
            _output_dir = re.sub(timestamp_ms_pattern, "{TIMESTAMP_MS}", _output_dir)
            _output_dir = re.sub(timestamp_pattern, "{TIMESTAMP}", _output_dir)
            _output_dir = re.sub(date_pattern, "{TIMESTAMP_DATE}", _output_dir)
            _output_dir = re.sub(time_pattern, "{TIMESTAMP_TIME}", _output_dir)
            return _output_dir

        return recover_output_dir(output_dir_1) == recover_output_dir(output_dir_2)


class PathConfig(BaseModel):
    """
    The output path of the logger and the callbacks are not included in the PathConfig.
    """

    exception_record_file_path: str
    config_output_path: str
    session_list_output_path: str
    metric_output_path: str
    coredumpy_output_dir: str
