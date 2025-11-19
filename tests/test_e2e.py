"""Testing End 2 End."""
from pathlib import Path
from moosez import moose

data = Path(__file__).parent / "data"

def test_basic_e2e(tmp_path) -> None:
    min_ct_throat = data / "CT_enhance_pet_1446_neck_slice.nii.gz"
    moose(input_data=str(min_ct_throat),
          model_names="clin_ct_organs",
          output_dir=tmp_path,
          accelerator="cpu")