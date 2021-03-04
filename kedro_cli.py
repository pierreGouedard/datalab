# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Command line tools for manipulating a Kedro project.

Intended to be invoked via `kedro`.
"""
# mypy: allow-untyped-defs, allow-untyped-calls, no-strict-optional, ignore-errors
import os
from itertools import chain
from pathlib import Path
from typing import Iterable, Tuple
import click
from kedro.framework.cli import main as kedro_main
from kedro.framework.cli.catalog import catalog as catalog_group
from kedro.framework.cli.jupyter import jupyter as jupyter_group
from kedro.framework.cli.pipeline import pipeline as pipeline_group
from kedro.framework.cli.project import project_group
from kedro.framework.cli.utils import split_string
from kedro.framework.context import load_context

# Prevent datadog to setup logs before kedro load log settings

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

# get our package onto the python path
PROJ_PATH = Path(__file__).resolve().parent

ENV_ARG_HELP = """Run the pipeline in a configured environment. If not specified,
pipeline will run using environment `local`."""
FROM_INPUTS_HELP = """A list of dataset names which should be used as a starting point."""
FROM_NODES_HELP = """A list of node names which should be used as a starting point."""
TO_NODES_HELP = """A list of node names which should be used as an end point."""
NODE_ARG_HELP = """Run only nodes with specified names."""
RUNNER_ARG_HELP = """Specify a runner that you want to run the pipeline with.
Available runners: `SequentialRunner`, `ParallelRunner` and `ThreadRunner`.
This option cannot be used together with --parallel."""
TAG_ARG_HELP = """Construct the pipeline using only nodes which have this tag
attached. Option can be used multiple times, what results in a
pipeline constructed from nodes having any of those tags."""
CONFIG_FILE_HELP = """Specify a YAML configuration file to load the run
command arguments from. If command line arguments are provided, they will
override the loaded ones."""
PIPELINE_ARG_HELP = """Name of the modular pipeline to run.
If not set, the project pipeline is run by default."""
PARAMS_ARG_HELP = """Specify extra parameters that you want to pass
to the context initializer. Items must be separated by comma, keys - by colon,
example: param1:value1,param2:value2. Each parameter is split by the first comma,
so parameter values are allowed to contain colons, parameter keys are not."""
ENV_HELP = """Environment name. Defaults to `staging.env`."""
MISSING_HELP = """Specify whether to run only nodes where output is missing"""
DEBUG_HELP = """Specify whether to run in debug mode (no DB save) nodes."""


def _get_values_as_tuple(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(chain.from_iterable(value.split(",") for value in values))


def _split_params(ctx, param, value):
    """
    Split params.

    params
    """
    if isinstance(value, dict):
        return value
    result = {}
    for item in split_string(ctx, param, value):
        item = item.split(":", 1)
        if len(item) != 2:
            ctx.fail(
                f"Invalid format of `{param.name}` option: Item `{item[0]}` must contain "
                f"a key and a value separated by `:`."
            )
        key = item[0].strip()
        if not key:
            ctx.fail(f"Invalid format of `{param.name}` option: Parameter key " f"cannot be an empty string.")
        value = item[1].strip()
        result[key] = _try_convert_to_numeric(value)
    return result


def _try_convert_to_numeric(value):
    """Try numeric conversion."""
    try:
        value = float(value)
    except ValueError:
        return value
    return int(value) if value.is_integer() else value


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""


@cli.command()
@click.option("--tag", "-t", type=str, multiple=True, help=TAG_ARG_HELP)
@click.option("--env", "-e", type=str, default="staging.env", help=ENV_HELP)
@click.option("--node", "-n", "node_names", type=str, multiple=True, help=NODE_ARG_HELP)
@click.option("--to-nodes", type=str, default="", help=TO_NODES_HELP, callback=split_string)
@click.option("--from-nodes", type=str, default="", help=FROM_NODES_HELP, callback=split_string)
@click.option("--from-inputs", type=str, default="", help=FROM_INPUTS_HELP, callback=split_string)
@click.option("--pipeline", type=str, default=None, help=PIPELINE_ARG_HELP)
@click.option("--debug", type=bool, multiple=False, default=False, help=DEBUG_HELP)
@click.option("--params", type=str, default="", help=PARAMS_ARG_HELP, callback=_split_params)
@click.option("--only-missing", "-m", type=bool, multiple=False, default=False, help=MISSING_HELP)
def run(tag, env, node_names, to_nodes, from_nodes, from_inputs, pipeline, debug, params, only_missing):
    """Run the pipeline."""
    tag = _get_values_as_tuple(tag) if tag else tag
    node_names = _get_values_as_tuple(node_names) if node_names else node_names
    context = load_context(Path.cwd(), env=env, extra_params=params, debug=debug, only_missing=only_missing)
    context.run(
        tags=tag,
        runner=context.runner,
        node_names=node_names,
        from_nodes=from_nodes,
        to_nodes=to_nodes,
        from_inputs=from_inputs,
        pipeline_name=pipeline,
    )


cli.add_command(pipeline_group)
cli.add_command(catalog_group)
cli.add_command(jupyter_group)

for command in project_group.commands.values():
    cli.add_command(command)


if __name__ == "__main__":
    os.chdir(str(PROJ_PATH))
    kedro_main()
