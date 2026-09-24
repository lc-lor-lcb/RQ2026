"""Bundle the current source and a matching saved policy for Colab.

This does not publish anything. Upload the resulting zip in Colab's Files pane
before running the notebook setup cell.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def package_walk(folder, output):
    folder, output = Path(folder), Path(output)
    required = ['walk_model.zip', 'walk_model_vecnorm.pkl', 'walk_params.json', 'evaluation.json']
    for name in required:
        if not (folder / name).is_file():
            raise FileNotFoundError(folder / name)
    required += [name for name in ['walk_curriculum.json', 'walk_heldout_evaluation.json',
                                  'walk_posture_curriculum.json', 'walk_smooth_curriculum.json',
                                  'walk_transition_evaluation.json']
                 if (folder/name).is_file()]
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for directory, pattern in [('roboquest', '*.py'), ('scripts', '*.py'), ('models/go2', '*')]:
        files.extend(p for p in (ROOT / directory).rglob(pattern)
                     if p.is_file() and '__pycache__' not in p.parts)
    files += [ROOT / name for name in ['requirements.txt', 'requirements-training.txt', 'setup.py',
        'notebooks/quickstart_ja.ipynb', 'notebooks/guide_and_experiments_ja.ipynb',
        'notebooks/saved_walk_viewer_ja.ipynb']]
    manifest = {}
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(set(files)):
            relative = file.relative_to(ROOT).as_posix()
            archive.write(file, 'RoboQuest2026/' + relative)
            manifest[relative] = hashlib.sha256(file.read_bytes()).hexdigest()
        for name in required:
            file = folder / name
            relative = 'models/teams/bundled_walk/' + name
            archive.write(file, 'RoboQuest2026/' + relative)
            manifest[relative] = hashlib.sha256(file.read_bytes()).hexdigest()
        archive.writestr('RoboQuest2026/bundle_manifest.json', json.dumps(manifest, indent=2))
    print(output.resolve())
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--output', default='artifacts/walk_colab_bundle.zip')
    args = parser.parse_args()
    package_walk(args.folder, args.output)
