"""Kedro orchestration entry point."""
# mypy: no-strict-optional
# Global import
from pathlib import Path
from kedro.framework.context import KedroContext
from kedro.pipeline import Pipeline
from kedro.io import DataCatalog
from kedro.runner.sequential_runner import SequentialRunner
from typing import Any, Dict, Union
from dotenv import load_dotenv
import logging

# Local import
from higgs import create_transform_pipeline  # type: ignore


class ProjectContext(KedroContext):
    """Implement ``KedroContext``."""

    project_name = "datalab-pipeline"
    project_version = "0.16.1"  # Kedro version
    package_name = "datalab"

    def __init__(
        self,
        project_path: Union[Path, str],
        env: str = "staging.env",
        extra_params: Dict[str, Any] = None,
        only_missing: bool = True,
        debug: bool = False,
    ):

        super(ProjectContext, self).__init__(project_path=project_path, extra_params=extra_params)

        # Add local env file variables to global variables
        load_dotenv(dotenv_path=self._project_path / "conf" / "local" / env)
        logging.info(f"Env loaded")

        # Instantiate the runner
        self.runner = JaneRunner(only_missing=only_missing)

    def _get_pipelines(self) -> Dict[str, Pipeline]:

        higgs_transform = create_transform_pipeline()

        return {
            "higgs": higgs_transform
        }


class JaneRunner(SequentialRunner):
    """
    ``IncrementalRunner`` is a ``SequentialRunner`` implementation.

    It can be used to run the ``Pipeline`` in a sequential manner using a topological sort of provided nodes. It also
    enable to filter out nodes where output already exists.
    """

    def __init__(self, only_missing: bool = True, is_async: bool = False):
        self.only_missing = only_missing
        super(JaneRunner, self).__init__(is_async=is_async)

    def run(self, pipeline: Pipeline, catalog: DataCatalog, run_id: str = None) -> Dict[str, Any]:
        """
        Run the ``Pipeline`` using the ``DataSet``s provided by ``catalog``.

        Parameters
        ----------
        pipeline: Pipeline
            The ``Pipeline`` to run
        catalog: DataCatalog
            The ``DataCatalog`` from which to fetch data.
        run_id: str
            The id of the run.

        Returns
        -------
        dict
            Any node outputs that cannot be processed by the ``DataCatalog``.
            These are returned in a dictionary, where the keys are defined
            by the node outputs.

        """
        # If missing flag run missing_output pipeline and its child nodes
        if self.only_missing:
            to_build = {ds for ds in catalog.list() if not catalog.exists(ds)}.intersection(pipeline.data_sets())
            pipeline = pipeline.only_nodes_with_outputs(*to_build) + pipeline.from_inputs(*to_build)

        return super(JaneRunner, self).run(pipeline, catalog, run_id)
