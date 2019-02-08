# Copyright 2018 Brendan Duke.
#
# This file is part of Mobile Prune.
#
# Mobile Prune is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Mobile Prune is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Mobile Prune. If not, see <http://www.gnu.org/licenses/>.

"""Experiment copy script."""
import json
import os
import re

import click
import path


@click.command()
@click.option('--from-log-dir',
              type=str,
              default=None,
              help='Log directory to copy experiment from')
@click.option('--to-log-dir',
              type=str,
              default=None,
              help='Log dir to copy experiment to')
def copy_experiment(from_log_dir, to_log_dir):
    """Copy config files and directory structure for one experiment to a new
    experiment.
    """
    from_log_dir = path.Path(from_log_dir)
    to_log_dir = path.Path(to_log_dir)

    os.system(f'mkdir -p {to_log_dir}')

    from_exp_num = os.path.basename(from_log_dir.rstrip('/'))
    to_exp_num = os.path.basename(to_log_dir.rstrip('/'))

    with open(from_log_dir/'config.json', 'r') as f:
        config_json = json.load(f)

    config_json['log_dir'] = str(to_log_dir)
    config_json['description'] = re.sub(
        f'{from_exp_num}$',
        f'{to_exp_num}',
        config_json['description'])
    config_json['log_file_path'] = str(to_log_dir/f'train.log')

    with open(to_log_dir/'config.json', 'w') as f:
        f.write(json.dumps(config_json, indent=8))


if __name__ == '__main__':
    copy_experiment()  # pylint:disable=no-value-for-parameter
