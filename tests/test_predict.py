import pytest
import numpy as np
import nibabel as nib

import moosez.resources
import moosez.predict
import moosez.constants
import moosez.file_utilities
import moosez.download
from moosez.benchmarking.profiler import Profiler


@pytest.fixture()
def setup_temp_dir(tmp_path_factory) -> tuple[str, str, str]:
    np.random.seed(0)

    tmp_root_dir = tmp_path_factory.mktemp(f"moosez-v{moosez.__version__}")
    image_path = tmp_root_dir.joinpath("CT")
    seg_path = tmp_root_dir.joinpath("segmentations")
    image_path.mkdir()
    seg_path.mkdir()

    data = np.random.randint(
        np.iinfo(np.int16).max, size=(500, 500, 390), dtype=np.int16)
    img = nib.Nifti1Image(dataobj=data, affine=np.eye(4))
    img.header.set_xyzt_units(xyz="mm")
    nib.save(img, image_path.joinpath("CT_XXXX-PETCT_XXXX-CT_0000.nii.gz"))

    return tmp_root_dir, image_path, seg_path


@pytest.mark.unit
def test_predict(setup_temp_dir):
    tmp_root_dir, input_dir, output_dir = setup_temp_dir
    accelerator = moosez.resources.check_device()
    model_name = "clin_ct_organs"
    model_path = moosez.constants.NNUNET_RESULTS_FOLDER
    label_prefix = moosez.resources.MODELS[model_name]['multilabel_prefix']

    Profiler.create_singleton_instance()
    moosez.file_utilities.create_directory(model_path)
    moosez.download.model(model_name, model_path)
    moosez.predict.predict(model_name, input_dir, output_dir, accelerator)
    Profiler.clear_singleton_instance()

    assert output_dir.joinpath("dataset.json").is_file() \
        and output_dir.joinpath("plans.json").is_file() \
        and output_dir.joinpath("predict_from_raw_data_args.json").is_file() \
        and sorted(output_dir.glob(label_prefix + "*"))[0].is_file()
