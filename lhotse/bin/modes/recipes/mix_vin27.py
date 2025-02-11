import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.mix_vin27 import prepare_mix_vin27
from lhotse.utils import Pathlike

__all__ = ["mix_vin27"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def mix_vin27(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
):
    """Mix_Vin27 data preparation."""
    prepare_mix_vin27(corpus_dir, output_dir=output_dir)


# @download.command(context_settings=dict(show_default=True))
# @click.argument("target_dir", type=click.Path(), default=".")
# def ljspeech(target_dir: Pathlike):
#     """LJSpeech download."""
#     download_ljspeech(target_dir)
