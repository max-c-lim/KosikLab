import json

CURATION_HISTORY_PATH = "data/waveforms/5116/curation_history.json"


def extract_curation_history(history_path):
    with open(history_path, "r") as f:
        history = json.load(f)

    unit_ids = set(history["initial"])
    print(f"N units before curation: {len(unit_ids)}")
    for curation in history["curations"]:
        unit_ids.intersection_update(history["curated"][curation])
        print(f"N units after {curation}: {len(unit_ids)}")


if __name__ == "__main__":
    extract_curation_history(CURATION_HISTORY_PATH)
