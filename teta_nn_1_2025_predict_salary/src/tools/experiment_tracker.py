"""
Модуль для работы с ClearML: инициализация экспериментов, логирование данных,
загрузка и сохранение артефактов.
Поддерживает конфигурацию через Hydra или прямое задание параметров.
"""

import sys
import logging
from typing import Optional, Dict, Any, Union, List, Final
from pathlib import Path

import pandas as pd

from omegaconf import DictConfig
from clearml import Task, Dataset, Model, OutputModel, TaskTypes

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

try:
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    logging.warning("ClearML не установлен. Установите с помощью: pip install clearml")


class ExperimentTracker:
    """
    Обертка для работы с ClearML для отслеживания экспериментов, логирования и работы с данными.
    Поддерживает конфигурацию через Hydra или прямую параметризацию.
    """

    task_types_clearml: Final[dict[str, TaskTypes]] = {
        "training": TaskTypes.training,
        "testing": TaskTypes.testing,
        "inference": TaskTypes.inference,
        "data_processing": TaskTypes.data_processing,
        "optimizer": TaskTypes.optimizer,
        "custom": TaskTypes.custom,
    }

    def __init__(
        self,
        project_name: str | None = None,
        task_name: str | None = None,
        task_type: str | None = None,
        config: Dict[str, Any] | DictConfig | None = None,
        use_clearml: bool = True,
        tags: List[str] | None = None,
        reuse_last_task_id: bool | None = False,
        output_uri: str | None = None,
    ):
        """
        Инициализация трекера экспериментов.

        Args:
            project_name: Название проекта в ClearML
            task_name: Название задачи/эксперимента. Если None, будет сгенерировано автоматически
            task_type: Тип задачи ('training', 'testing', 'inference', 'data_processing')
            config: Конфигурация эксперимента (словарь или DictConfig из Hydra)
            use_clearml: Флаг использования ClearML (можно отключить для локальных тестов)
            tags: Теги для задачи
            reuse_last_task_id: Переиспользовать последний Task ID (для продолжения эксперимента)
            output_uri: URI для сохранения моделей и артефактов (опционально)
        """
        self.config = config
        self.use_clearml = use_clearml and CLEARML_AVAILABLE
        self.task = None
        self.logger = None

        task_config = {}
        if config:
            if hasattr(config, "get"):
                task_config = config.get("task", {})
            elif isinstance(config, dict):
                task_config = config.get("task", {})

        if self.use_clearml:
            self.task = Task.init(
                project_name=project_name or task_config.get("project_name"),
                task_name=task_name or task_config.get("task_name"),
                task_type=self.task_types_clearml[
                    task_type or task_config.get("task_type", "custom")
                ],
                reuse_last_task_id=reuse_last_task_id
                if reuse_last_task_id is not None
                else task_config.get("reuse_last_task_id", False),
                output_uri=output_uri or task_config.get("output_uri"),
                tags=tags or task_config.get("tags", []),
            )

            if config:
                self.task.connect(config)

            self.logger = self.task.get_logger()
            logging.info(f"[CLEARML] Task initialized with ID: {self.task.id}")
        else:
            logging.info(
                "[LOCAL] ClearML is disabled - running in local mode without cloud storage"
            )

    def log_table(
        self,
        title: str,
        series: str,
        table: pd.DataFrame,
        iteration: int | None = None,
    ) -> None:
        """
        Логирование таблиц.

        Args:
            title: Название таблицы
            series: Название серии данных
            table: DataFrame для логирования
            iteration: Итерация (шаг)
        """
        if self.use_clearml and self.logger is not None:
            self.logger.report_table(
                title=title, series=series, iteration=iteration, table_plot=table
            )
            logging.info(f"[CLEARML] Table '{title}' logged with shape {table.shape}")
        else:
            logging.warning(
                f"[LOCAL] ClearML is disabled - table '{title}' not saved to cloud storage"
            )

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """
        Логирование гиперпараметров.

        Args:
            params: Словарь с гиперпараметрами
        """
        if self.use_clearml and self.task:
            self.task.connect(params)
            logging.info(
                f"[CLEARML] Hyperparameters connected: {len(params)} parameters"
            )
        else:
            logging.warning(
                f"[LOCAL] ClearML is disabled - hyperparameters ({len(params)} params) not saved to cloud storage"
            )

    def log_artifact(
        self, name: str, artifact: Any, upload_uri: str | None = None
    ) -> None:
        """
        Сохранение артефакта.

        Args:
            name: Название артефакта
            artifact: Объект для сохранения
            upload_uri: URI для загрузки артефакта (опционально)
        """
        if self.use_clearml and self.task:
            self.task.upload_artifact(
                name=name, artifact=artifact, upload_uri=upload_uri
            )
            logging.info(f"[CLEARML] Artifact '{name}' uploaded successfully")
        else:
            logging.warning(
                f"[LOCAL] ClearML is disabled - artifact '{name}' not saved to cloud storage"
            )

    def log_file(self, name: str, file_path: Union[str, Path]) -> None:
        """
        Сохранение файла как артефакта.

        Args:
            name: Название артефакта
            file_path: Путь к файлу
        """
        if self.use_clearml and self.task:
            self.task.upload_artifact(name=name, artifact_object=Path(file_path))
            logging.info(f"[CLEARML] File '{name}' uploaded from: {file_path}")
        else:
            logging.warning(
                f"[LOCAL] ClearML is disabled - file '{name}' not uploaded to cloud storage"
            )

    def get_dataset(
        self,
        dataset_id: str,
    ) -> Optional[str]:
        """
        Получение датасета из ClearML.

        Args:
            dataset_id: ID датасета

        Returns:
            Путь к локальной копии датасета или None при ошибке
        """
        if not self.use_clearml:
            logging.error(
                "[LOCAL] ClearML is disabled - cannot download dataset from cloud storage"
            )
            return None

        try:
            dataset = Dataset.get(dataset_id=dataset_id)
            local_path = dataset.get_local_copy()

            logging.info(f"[CLEARML] Dataset downloaded to: {local_path}")
            return local_path
        except Exception as e:
            logging.error(f"[CLEARML] Error downloading dataset: {e}")
            return None

    def save_model(
        self, name: str, model_path: str, tags: List[str] | None = None
    ) -> Optional[str]:
        """
        Сохранение модели.

        Args:
            name: Название модели
            model_path: Путь к файлу модели
            tags: Теги модели

        Returns:
            ID модели в ClearML или None при ошибке
        """
        if not self.use_clearml:
            logging.warning(
                f"[LOCAL] ClearML is disabled - model '{name}' not saved to cloud storage"
            )
            return None

        try:
            output_model = OutputModel(task=self.task, name=name, tags=tags)
            output_model.update_weights(weights_filename=model_path)

            logging.info(f"[CLEARML] Model '{name}' saved with ID: {output_model.id}")
            return output_model.id
        except Exception as e:
            logging.error(f"[CLEARML] Error saving model: {e}")
            return None

    def load_model(self, model_id: str) -> Any:
        """
        Загрузка модели из ClearML.

        Args:
            model_id: ID модели

        Returns:
            Путь к загруженной модели или None при ошибке
        """
        if not self.use_clearml:
            logging.error(
                "[LOCAL] ClearML is disabled - cannot load model from cloud storage"
            )
            return None

        try:
            model = Model(model_id=model_id)
            local_weights_path = model.get_local_copy()

            logging.info(f"[CLEARML] Model downloaded to: {local_weights_path}")
            return local_weights_path
        except Exception as e:
            logging.error(f"[CLEARML] Error loading model: {e}")
            return None

    def close(self) -> None:
        """
        Завершение эксперимента.
        """
        if self.use_clearml and self.task is not None:
            logging.info(f"[CLEARML] Closing task: {self.task.id}")
            self.task.close()
        else:
            logging.warning("[LOCAL] ClearML is disabled")


def create_experiment_from_hydra(cfg: DictConfig) -> ExperimentTracker:
    """
    Создание трекера экспериментов из конфигурации Hydra.

    Args:
        cfg: Конфигурация Hydra

    Returns:
        Инициализированный трекер экспериментов
    """
    tracking_cfg = cfg.get("tracking", {})

    return ExperimentTracker(
        project_name=tracking_cfg.get("project_name"),
        task_name=tracking_cfg.get("task_name"),
        task_type=tracking_cfg.get("task_type", "training"),
        config=cfg,
        use_clearml=tracking_cfg.get("use_clearml", True),
        tags=tracking_cfg.get("tags"),
        reuse_last_task_id=tracking_cfg.get("reuse_last_task_id", False),
        output_uri=tracking_cfg.get("output_uri"),
    )
