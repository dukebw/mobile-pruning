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
