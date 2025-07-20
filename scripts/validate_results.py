import json
from pathlib import Path
from jsonschema import validate

SCHEMA_PATH = Path('results/results-schema.json')
SCHEMA = json.loads(SCHEMA_PATH.read_text())

for path in Path('results').glob('*.json'):
    if path.name == 'results-schema.json':
        continue
    data = json.loads(path.read_text())
    validate(instance=data, schema=SCHEMA)
    print(f"Validated {path}")
