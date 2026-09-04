from pathlib import Path
import subprocess
import sys


SCRIPTS = [
    "01_download_data.py",
    "02_compute_returns.py",
    "03_build_windows.py",
    "04_estimate_parameters.py",
    "05_compute_real_topology.py",
    "06_compute_surrogates.py",
    "07_build_envelopes.py",
    "08_generate_outputs.py",
]


def main():
    project_root = Path(__file__).resolve().parent
    scripts_dir = project_root / "scripts"

    for script in SCRIPTS:
        script_path = scripts_dir / script

        if not script_path.exists():
            raise FileNotFoundError(f"Script não encontrado: {script_path}")

        print(f"\n=== Executando {script} ===", flush=True)

        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_root,
            check=True,
        )

    print("\nPipeline concluído com sucesso.", flush=True)


if __name__ == "__main__":
    main()