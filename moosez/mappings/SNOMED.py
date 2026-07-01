"""
moosez label -> standardized coding for DICOM Segmentation (DICOM SEG).

The mapping is sourced from the curated table ``moose_snomed_mapping.csv`` that
ships next to this module, and is exposed as the module-level dictionary
``moose_to_snomed`` (keyed by moosez label name).

Compared with the previous hand-maintained ``{"ID", "name"}`` dictionary, each
entry now carries the full set of attributes a valid DICOM SEG segment needs
(see ENHANCE-PET/MOOSE#240):

* segmented property category
* segmented property type
* segmented property type modifier (e.g. laterality -- so left/right structures
  that share a type code are no longer indistinguishable)
* anatomic region (and its modifier)
* recommended display RGB color

Each value is a dict whose keys match the DICOM attribute names used by dcmqi's
``itkimage2segimage`` metadata, so an entry can be dropped almost verbatim into a
dcmqi ``segmentAttributes`` record::

    moose_to_snomed["iliac_vena_left"] == {
        "SegmentedPropertyCategoryCodeSequence":
            {"CodingSchemeDesignator": "SCT", "CodeValue": "91723000",
             "CodeMeaning": "Anatomical Structure"},
        "SegmentedPropertyTypeCodeSequence":
            {"CodingSchemeDesignator": "SCT", "CodeValue": "46027005",
             "CodeMeaning": "Common iliac vein"},
        "SegmentedPropertyTypeModifierCodeSequence":
            {"CodingSchemeDesignator": "SCT", "CodeValue": "7771000",
             "CodeMeaning": "Left"},
        "AnatomicRegionSequence": None,
        "AnatomicRegionModifierSequence": None,
        "recommendedDisplayRGBValue": [197, 174, 37],
    }

Optional code sequences that are absent for a given label are present with a
value of ``None`` so the schema is uniform across entries.
"""

import csv
import importlib.resources

__all__ = ["moose_to_snomed", "load_mapping", "CSV_FILENAME"]

CSV_FILENAME = "moose_snomed_mapping.csv"

# Code sequences encoded in the CSV; each spans three columns sharing a prefix
# (.CodingSchemeDesignator / .CodeValue / .CodeMeaning).
_CODE_SEQUENCES = (
    "SegmentedPropertyCategoryCodeSequence",
    "SegmentedPropertyTypeCodeSequence",
    "SegmentedPropertyTypeModifierCodeSequence",
    "AnatomicRegionSequence",
    "AnatomicRegionModifierSequence",
)


def _code(row, prefix):
    """Build a {Designator, Value, Meaning} dict from a CSV row, or None if empty."""
    value = (row.get(f"{prefix}.CodeValue") or "").strip()
    if not value:
        return None
    return {
        "CodingSchemeDesignator": (row.get(f"{prefix}.CodingSchemeDesignator") or "").strip(),
        "CodeValue": value,
        "CodeMeaning": (row.get(f"{prefix}.CodeMeaning") or "").strip(),
    }


def _rgb(raw):
    """Parse a "[r, g, b]" cell into [int, int, int], or None if absent/invalid."""
    raw = (raw or "").strip().strip("[]")
    if not raw:
        return None
    try:
        values = [int(part) for part in raw.split(",")]
    except ValueError:
        return None
    return values if len(values) == 3 else None


def load_mapping():
    """Load and return the label -> coding mapping from the bundled CSV.

    The CSV has one row per (model, label); the returned dict is keyed by label
    name. Where a label is shared by several models its coding is identical, so a
    flat label-keyed dict is unambiguous; a genuine conflict raises ValueError.
    """
    text = importlib.resources.files(__package__).joinpath(CSV_FILENAME).read_text(encoding="utf-8")

    mapping = {}
    for row in csv.DictReader(text.splitlines()):
        label = (row.get("label_name") or "").strip()
        if not label:
            continue
        entry = {prefix: _code(row, prefix) for prefix in _CODE_SEQUENCES}
        entry["recommendedDisplayRGBValue"] = _rgb(row.get("recommendedDisplayRGBValue"))

        if label in mapping and mapping[label] != entry:
            raise ValueError(
                f"Conflicting SNOMED coding for label '{label}' in {CSV_FILENAME}"
            )
        mapping[label] = entry

    return mapping


moose_to_snomed = load_mapping()
