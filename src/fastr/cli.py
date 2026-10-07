"""fastr - an application to rapidly predict TCR:pMHC structures using AlphaFold 3."""

import argparse
import hashlib
import logging
import os
from pathlib import Path

from fastr._log import add_logging_arguments, setup_logger
from fastr.af3_predictions import run_alphafold3_predictions
from fastr.msas_and_templates import create_msas_and_templates

parser = argparse.ArgumentParser()

logger = logging.getLogger()

parser = argparse.ArgumentParser(
    prog='fastr',
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

input_group = parser.add_argument_group('Input')
input_group.add_argument('--tcra', required=True, help='sequence of TCR alpha chain')
input_group.add_argument('--tcrb', required=True, help='sequence of TCR beta chain')
input_group.add_argument('--peptide', required=True, help='sequence of peptide')
input_group.add_argument('--mhca', required=True, help='sequence of MHC alpha chain')
input_group.add_argument('--mhcb', required=True, help='sequence of MHC beta chain')

output_group = parser.add_argument_group('Output')
output_group.add_argument(
    '--output', '-o', required=True, help='path to the output directory with the prediction files inside'
)

parser.add_argument('--num-threads', default=32, type=int, help='number of threads to use to create MSAs')
parser.add_argument(
    '--temp-dir', default=None, type=str, help='path to temporary directory to store between run cache (Default: None)'
)

prediction_parameters = parser.add_argument_group('Prediction parameters')
prediction_parameters.add_argument('--name', default='prediction', help='name to give the output prediction')
prediction_parameters.add_argument('--num-recycles', type=int, default=10, help='number of recycles to run')
prediction_parameters.add_argument('--seeds', nargs='+', default=[1], help='random seeds to use for predictions')

add_logging_arguments(parser)


def create_name(sequence: str) -> str:
    """Create entry name for a sequence by hashing."""
    return hashlib.sha256(sequence.encode()).hexdigest()[:8]


def get_cached_names(unpaired_msa_dir: Path, paired_msa_dir: Path, templates_dir: Path) -> set:
    """Get all of the precompute MSA and template names."""
    unpaired_msa_names = {name.stem for name in unpaired_msa_dir.iterdir()}
    paired_msa_names = {name.stem for name in paired_msa_dir.iterdir()}
    template_names = {name.name for name in templates_dir.iterdir()}

    return unpaired_msa_names & paired_msa_names & template_names


def main() -> None:
    """Run the application."""
    args = parser.parse_args()
    setup_logger(logger, args.log_level, args.log_file)

    base_dir = Path(os.environ['FASTR_BASE_DIR'])

    alphafold_data_dir = Path(os.environ['TCRMODEL2_ALPHAFOLD_DATA_DIR'])
    uniref90_database_path = '${DB_DIR}/uniref90.tcrmhc.fasta'
    mgnify_database_path = '${DB_DIR}/mgnify.fasta'
    small_bfd_database_path = '${DB_DIR}/small_bfd.tcrmhc.fasta'
    uniprot_cluster_annot_database_path = '${DB_DIR}/uniprot.tcrmhc.fasta'
    seqres_database_path = '${DB_DIR}/pdb_seqres.txt'
    pdb_database_path = '${DB_DIR}/pdb_mmcif/mmcif_files'

    cache_dir = base_dir / 'cache'
    cache_dir.mkdir(exist_ok=True)

    unpaired_msas_dir = cache_dir / 'unpaired_msas'
    unpaired_msas_dir.mkdir(exist_ok=True)

    paired_msas_dir = cache_dir / 'paired_msas'
    paired_msas_dir.mkdir(exist_ok=True)

    templates_dir = cache_dir / 'templates'
    templates_dir.mkdir(exist_ok=True)

    cached_names = get_cached_names(unpaired_msas_dir, paired_msas_dir, templates_dir)

    names = {}
    sequences = {}
    unpaired_msa_paths = {}
    paired_msa_paths = {}
    template_paths = {}

    for chain_type, sequence in ('TCRa', args.tcra), ('TCRb', args.tcrb), ('MHCa', args.mhca), ('MHCb', args.mhcb):
        names[chain_type] = create_name(sequence)
        sequences[chain_type] = sequence

        if names[chain_type] not in cached_names:
            logger.info('Creating MSAs for %s', chain_type)
            create_msas_and_templates(
                names[chain_type],
                sequence,
                base_dir=base_dir,
                alphafold_data_dir=alphafold_data_dir,
                uniref90_database=uniref90_database_path,
                mgnify_database=mgnify_database_path,
                small_bfd_database=small_bfd_database_path,
                uniprot_cluster_annot_database=uniprot_cluster_annot_database_path,
                seqres_database=seqres_database_path,
                pdb_database=pdb_database_path,
                output_dir=cache_dir,
                num_threads=args.num_threads,
            )

        unpaired_msa_paths[chain_type] = unpaired_msas_dir / (names[chain_type] + '.a3m')
        paired_msa_paths[chain_type] = paired_msas_dir / (names[chain_type] + '.a3m')
        template_paths[chain_type] = templates_dir / names[chain_type]

    sequences['peptide'] = args.peptide

    logger.info('Running structure prediction for %s', args.name)
    run_alphafold3_predictions(
        args.name,
        Path(args.output),
        sequences,
        num_recycles=args.num_recycles,
        seeds=args.seeds,
        base_dir=base_dir,
        unpaired_msa_paths=unpaired_msa_paths,
        paired_msa_paths=paired_msa_paths,
        template_paths=template_paths,
        temp_dir=Path(args.temp_dir) if args.temp_dir is not None else None,
    )


if __name__ == '__main__':
    main()
