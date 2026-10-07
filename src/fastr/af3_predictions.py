"""Predict structure with AlphaFold3."""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def _fill_templates(template_paths: dict[Path], *, relative_paths: bool = True) -> dict[str, list[dict[str, str]]]:
    templates = {}
    for chain_type, path in template_paths.items():
        template_entries = defaultdict(dict)
        indices_paths = path.glob('*_indices.json')
        structure_paths = path.glob('*.cif')

        for indice_path in indices_paths:
            entry_name = re.search(r'(\d+)_indices.json', indice_path.name).group(1)
            template_entries[entry_name]['indices_path'] = indice_path

        for structure_path in structure_paths:
            entry_name = re.search(r'(\d+).cif', structure_path.name).group(1)
            template_entries[entry_name]['structure_path'] = structure_path

        template_entries = sorted(template_entries.items())

        entry_templates = []
        for _, entry in template_entries:
            with open(entry['indices_path'], 'r') as fh:
                entry_template = json.load(fh)

            if relative_paths:
                entry_template['mmcifPath'] = str(entry['structure_path']).replace(str(path.parent), '').strip('/')

            else:
                entry_template['mmcifPath'] = entry['structure_path']

            entry_templates.append(entry_template)

        templates[chain_type] = entry_templates

    return templates


def format_sequences_for_af3(
    name: str,
    sequences: dict[str, str],
    output_dir: Path,
    seeds: list[int],
    *,
    add_paths: bool = False,
    relative_paths: bool = True,
    unpaired_msa_paths: dict[Path] | None = None,
    paired_msa_paths: dict[Path] | None = None,
    template_paths: dict[Path] | None = None,
) -> None:
    """Take sequences and MSA paths and create a json file in the specified location.

    Args:
        name: name to give to the prediction
        sequences: sequences stored in a dictionary and identified as TCRa, TCRb, peptide, MHCa, or MHCb
        output_dir: path to write the prediction too
        seeds: random seeds to use for prediction (Default: [1])
        add_paths: add paths for the MSAs and templates(Default: True),
        relative_paths: use relative paths for the MSAs and templates (Default: True),
        unpaired_msa_paths: list of paths for the unpaired MSAs to use
        paired_msa_paths: list of paths for the paired MSAs to use
        template_paths: list of paths for the templates to use

    """
    if unpaired_msa_paths is None:
        unpaired_msa_paths = {}

    if paired_msa_paths is None:
        paired_msa_paths = {}

    template_paths = (
        _fill_templates(template_paths, relative_paths=relative_paths) if template_paths is not None else {}
    )

    output_name = output_dir / f'{name}.json'

    chain_id_mapping = {'D': 'TCRa', 'E': 'TCRb', 'C': 'peptide', 'A': 'MHCa', 'B': 'MHCb'}
    output = {
        'name': name,
        'modelSeeds': seeds,
        'sequences': [
            {'protein': {'id': sequence_id, 'sequence': sequences[sequence_type]}}
            for sequence_id, sequence_type in chain_id_mapping.items()
            if sequence_type in sequences
        ],
        'dialect': 'alphafold3',
        'version': 2,
    }

    if add_paths:
        new_sequence_entries = []
        for sequence_entry in output['sequences']:
            new_sequence_entry = {**sequence_entry}

            if sequence_entry['protein']['id'] == 'C':
                msa = f">query\n{sequence_entry['protein']['sequence']}\n"
                new_sequence_entry['protein']['unpairedMsa'] = msa
                new_sequence_entry['protein']['pairedMsa'] = msa
                new_sequence_entry['protein']['templates'] = []

            else:
                chain_type = chain_id_mapping[sequence_entry['protein']['id']]

                new_sequence_entry['protein']['unpairedMsaPath'] = str(
                    unpaired_msa_paths[chain_type].stem + '_unpaired.a3m'
                    if relative_paths
                    else unpaired_msa_paths[chain_type]
                )
                new_sequence_entry['protein']['pairedMsaPath'] = str(
                    paired_msa_paths[chain_type].stem + '_paired.a3m'
                    if relative_paths
                    else paired_msa_paths[chain_type]
                )
                new_sequence_entry['protein']['templates'] = template_paths[chain_type]

            new_sequence_entries.append(new_sequence_entry)

        output['sequences'] = new_sequence_entries

    with open(output_name, 'w') as output_fh:
        logger.debug('Writing to %s', output_name)
        json.dump(output, output_fh)


def _create_command(
    *,
    name: str,
    base_dir: Path,
    input_dir_abs: Path,
    temp_output_dir_abs: Path,
    jax_cache_dir: Path | None,
    num_recycles: int,
) -> list[str]:
    cwd = os.getcwd()
    cmd = [
        '/usr/bin/apptainer',
        'exec',
        '--nv',
        f'--cwd={cwd}',
        f'--bind={cwd}:{cwd}',
        f'--bind={input_dir_abs}:/root/input',
        f'--bind={temp_output_dir_abs}:/root/output',
        f"--bind={os.environ['ALPHAFOLD3_WEIGHTS_DIR']}:/root/models",
    ]

    if jax_cache_dir:
        cmd.append(f"--bind={os.path.join(os.environ['TMPDIR'], 'jax_cache')}:/root/jax_cache")

    cmd.extend(
        [
            str(base_dir / 'alphafold3.sif'),
            'python',
            'third_party/alphafold3/run_alphafold.py',
            '--norun_data_pipeline',
            f'--num_recycles={num_recycles}',
            f'--json_path=/root/input/{name}.json',
            '--model_dir=/root/models',
            '--output_dir=/root/output',
        ]
    )

    if jax_cache_dir:
        cmd.append('--jax_compilation_cache_dir=/root/jax_cache')

    return cmd


def run_alphafold3_predictions(
    name: str,
    output_dir: Path,
    sequences: dict[str, str],
    *,
    base_dir: Path,
    unpaired_msa_paths: dict[Path],
    paired_msa_paths: dict[Path],
    template_paths: dict[Path],
    num_recycles: int = 10,
    seeds: list | None = None,
    temp_dir: Path | None = None,
) -> None:
    """Predict structure of TCR:pMHC using AF3.

    Args:
        name: name to give the prediction
        output_dir: path to write the prediction too
        sequences: sequences stored in a dictionary and identified as TCRa, TCRb, peptide, MHCa, or MHCb
        base_dir: base data directory for fastr
        unpaired_msa_paths: list of paths for the unpaired MSAs to use
        paired_msa_paths: list of paths for the paired MSAs to use
        template_paths: list of paths for the templates to use
        num_recycles: number of recycles to run (Default: 10)
        seeds: random seeds to use for prediction (Default: [1])
        temp_dir: temporary directory to store between run caches (Default: None)

    """
    if seeds is None:
        seeds = [1]

    if temp_dir is not None:
        jax_cache_dir = temp_dir / 'jax_cache'
        jax_cache_dir.mkdir(exist_ok=True)

    else:
        jax_cache_dir = None

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)

        input_dir_abs = tmp_dir / name / 'input'
        temp_output_dir_abs = tmp_dir / name / 'output'

        (tmp_dir / name).mkdir()
        input_dir_abs.mkdir()
        temp_output_dir_abs.mkdir()

        format_sequences_for_af3(
            name,
            sequences,
            input_dir_abs,
            seeds,
            add_paths=True,
            unpaired_msa_paths=unpaired_msa_paths,
            paired_msa_paths=paired_msa_paths,
            template_paths=template_paths,
        )

        for msa_path in unpaired_msa_paths.values():
            shutil.copy(str(msa_path), str(input_dir_abs / (msa_path.stem + '_unpaired.a3m')))

        for msa_path in paired_msa_paths.values():
            shutil.copy(str(msa_path), str(input_dir_abs / (msa_path.stem + '_paired.a3m')))

        for template_dir in template_paths.values():
            directory_name = template_dir.name
            new_directory_path = input_dir_abs / directory_name
            new_directory_path.mkdir()

            for template in template_dir.glob('*.cif'):
                shutil.copy(template, str(new_directory_path / template.name))

        subprocess.Popen(['/usr/bin/chmod', '+x', '-R', input_dir_abs])  # noqa: S603

        cmd = _create_command(
            name=name,
            base_dir=base_dir,
            input_dir_abs=input_dir_abs,
            temp_output_dir_abs=temp_output_dir_abs,
            jax_cache_dir=jax_cache_dir,
            num_recycles=num_recycles,
        )
        subprocess.run(cmd, check=True)  # noqa: S603

        output_dir.mkdir(exist_ok=True)

        sanitized_output_name = name.lower().replace(' ', '_')
        for seed in seeds:
            for sample_num in range(5):
                prediction_dir = temp_output_dir_abs / sanitized_output_name / f'seed-{seed}_sample-{sample_num}'

                shutil.copy(
                    str(prediction_dir / f'{sanitized_output_name}_seed-{seed}_sample-{sample_num}_model.cif'),
                    str(output_dir / f'{name}_seed-{seed}_sample-{sample_num}_model.cif'),
                )

                shutil.copy(
                    str(
                        prediction_dir
                        / f'{sanitized_output_name}_seed-{seed}_sample-{sample_num}_summary_confidences.json'
                    ),
                    str(output_dir / f'{name}_seed-{seed}_sample-{sample_num}_summary_confidences.json'),
                )

                shutil.copy(
                    str(prediction_dir / f'{sanitized_output_name}_seed-{seed}_sample-{sample_num}_confidences.json'),
                    str(output_dir / f'{name}_seed-{seed}_sample-{sample_num}_confidences.json'),
                )
