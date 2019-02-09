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

"""Update param in config file."""
import json

import click


@click.command()
@click.option('--log-dir', type=str, default=None, help='Log directory.')
@click.option('--key', type=str, default=None, help='Key.')
@click.option('--val', type=str, default=None, help='Value.')
@click.option('--val-type',
              type=click.Choice(['bool', 'float', 'int', 'string']))
def update_config(log_dir, key, val, val_type):
    """Reset a single parameter in config file."""
    name_to_type = {'bool': bool, 'float': float, 'int': int, 'string': str}
    val = name_to_type[val_type](val)

    fpath = f'{log_dir}/config.json'
    with open(fpath, 'r') as f:
        config_json = json.load(f)

    assert isinstance(val, type(config_json[key]))

    config_json[key] = val

    with open(fpath, 'w') as f:
        f.write(json.dumps(config_json, indent=8))


if __name__ == '__main__':
    update_config()  # pylint:disable=no-value-for-parameter
