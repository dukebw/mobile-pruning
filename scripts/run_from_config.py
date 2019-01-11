"""Calls script, configured by a configuration file."""
import re
import os
import json

import click
import path


@click.command()
@click.option('--config-dir',
              default=None,
              type=str,
              help='Path to config directory with config files. Config files '
                   'should be named: common_config.json, train_config.json, '
                   'and val_config.json')
@click.option('--gpu-devices',
              default=None,
              type=str,
              help='A comma-separated list of which GPU devices should be '
                   'visible, or the string CPU for CPU only.')
@click.option('--script-name',
              default=None,
              type=str,
              help='E.g., the string inference.train for training '
                   'script-name is the name of the Python script that will '
                   'be executed, hence train for train.py.')
def run_from_config(config_dir, gpu_devices, script_name):
    """Call a script using parameters from an input JSON config file.

    CUDA_VISIBLE_DEVICES is set using the gpu_devices parameter passed to this
    Python script.
    """
    config_dir = path.Path(config_dir)

    with open(config_dir/'config.json', 'r') as f:
        config = json.load(f)

    config_string = ""
    for option in config:
        # NOTE(brendan): Booleans are passed as --no-<option-name> for False,
        # and --<option-name> for True.
        if isinstance(config[option], bool):
            yes_no = '' if config[option] else 'no-'
            config_string += ' --' + yes_no + option
        elif isinstance(config[option], list):
            config_string += ' --' + option
            for thing in config[option]:
                config_string += ' ' + str(thing)
        elif config[option] is not None:
            config_string += ' --' + option + ' ' + str(config[option])

    cuda_visible_devices = 'CUDA_VISIBLE_DEVICES='
    if gpu_devices == 'CPU':
        gpu_devices = ''
    else:
        gpu_warn = 'gpu_devices should be of the format <int>(,<int>)*'
        assert (re.match(r'^\d(,\d)*$', gpu_devices) is not None), gpu_warn

    cuda_visible_devices += gpu_devices

    if script_name.endswith('.py'):
        cmd = f'{cuda_visible_devices} python3 {script_name} {config_string}'
    else:
        cmd = f'{cuda_visible_devices} python3 -m {script_name} {config_string}'
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    run_from_config()  # pylint:disable=no-value-for-parameter
