"""Create MSAs and find templates for a sequence."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

NUM_PARALLEL_AF_PROCESSES = 4
'''AF3 runs 4 processes at once for building the MSAs.'''


def _create_msa_input(name: str, sequence: str, output_dir: Path) -> None:
    with open(output_dir / (name + '.json'), 'w') as output_fh:
        json.dump(
            {
                'name': name,
                'modelSeeds': [1],
                'sequences': [
                    {'protein': {'id': 'A', 'sequence': sequence}},
                ],
                'dialect': 'alphafold3',
                'version': 1,
            },
            output_fh,
        )


def create_msas_and_templates(
    name: str,
    sequence: str,
    *,
    base_dir: Path,
    alphafold_data_dir: Path,
    uniref90_database: Path,
    mgnify_database: Path,
    small_bfd_database: Path,
    uniprot_cluster_annot_database: Path,
    seqres_database: Path,
    pdb_database: Path,
    output_dir: Path,
    num_threads: int = 32,
) -> None:
    """Create MSAs and templates for an amino acid."""
    cwd = os.getcwd()

    with tempfile.TemporaryDirectory() as temp_dir:
        _create_msa_input(name, sequence, Path(temp_dir))

        subprocess.run(  # noqa: S603
            [
                '/usr/bin/apptainer',
                'exec',
                f'--cwd={cwd}',
                f'--bind={cwd}:{cwd}',
                f'--bind={alphafold_data_dir}:/root/public_databases',
                f"--bind={str(pdb_database).replace('${DB_DIR}', str(alphafold_data_dir))}:/root/pdb",
                str(base_dir / 'alphafold3.sif'),
                'python',
                'third_party/alphafold3/run_alphafold.py',
                '--norun_inference',
                f'--jackhmmer_n_cpu={max(num_threads // NUM_PARALLEL_AF_PROCESSES, 1)}',
                f'--nhmmer_n_cpu={max(num_threads // NUM_PARALLEL_AF_PROCESSES, 1)}',
                '--db_dir=/root/public_databases',
                '--pdb_database_path=/root/pdb',
                f'--uniref90_database_path={uniref90_database}',
                f'--mgnify_database_path={mgnify_database}',
                f'--small_bfd_database_path={small_bfd_database}',
                f'--uniprot_cluster_annot_database_path={uniprot_cluster_annot_database}',
                f'--seqres_database_path={seqres_database}',
                f"--json_path={Path(temp_dir) / (name + '.json')}",
                f'--output_dir={temp_dir}',
            ],
            check=True,
        )

        unpaired_msas = output_dir / 'unpaired_msas'
        paired_msas = output_dir / 'paired_msas'
        templates = output_dir / 'templates'

        unpaired_msas.mkdir(parents=True, exist_ok=True)
        paired_msas.mkdir(parents=True, exist_ok=True)
        templates.mkdir(parents=True, exist_ok=True)

        with open(os.path.join(temp_dir, name, f'{name}_data.json'), 'r') as fh:
            msa_data = json.load(fh)

            with open(str(unpaired_msas / f'{name}.a3m'), 'w') as out_fh:
                out_fh.write(msa_data['sequences'][0]['protein']['unpairedMsa'])

            with open(str(paired_msas / f'{name}.a3m'), 'w') as out_fh:
                out_fh.write(msa_data['sequences'][0]['protein']['pairedMsa'])

            entry_templates = templates / name
            entry_templates.mkdir()
            for i, template in enumerate(msa_data['sequences'][0]['protein']['templates']):
                with open(str(entry_templates / f'{i}.cif'), 'w') as out_fh:
                    out_fh.write(template['mmcif'])

                with open(str(entry_templates / f'{i}_indices.json'), 'w') as out_fh:
                    json.dump({key: template[key] for key in ('queryIndices', 'templateIndices')}, out_fh)
