import pytest
import shutil
import pathlib
from distutils.dir_util import copy_tree
from datetime import datetime

import numpy as np
import moosez
from moosez.moosez import main
from monai.transforms import LoadImage
from moosez.resources import MODELS


CURR_DATE = datetime.now().strftime("%H-%M-%d-%m-%Y")
REFERENCE_PATH = pathlib.Path().joinpath("data", "external", "CT-images_N20_reference_output.zip")
OUTPUT_DIR = pathlib.Path().joinpath("data", "processed")
MODEL_NAME = "clin_ct_organs"


@pytest.fixture(scope="session")
def setup_prediction(tmp_path_factory) -> tuple[str, str, str]:
    tmp_root_dir = tmp_path_factory.mktemp(f"moosez-v{moosez.__version__}_")
    try:
        #TODO: better way to check correct reference data ?
        assert REFERENCE_PATH.stat().st_size == 516289595
    except AssertionError:
        raise Exception("Incorrect reference data!")

    
    shutil.unpack_archive(REFERENCE_PATH.resolve(), tmp_root_dir)

    input_image_path = tmp_root_dir.joinpath("CT-images_N20_reference_output", "input", "images")
    output_image_path = tmp_root_dir.joinpath("CT-images_N20_reference_output", "output", "images")

    main(["-m", MODEL_NAME, "-d", str(input_image_path)])
    copy_tree(str(input_image_path.resolve()), str(OUTPUT_DIR.joinpath(CURR_DATE, "images").resolve()))

    return input_image_path, output_image_path


@pytest.mark.non_reg
@pytest.mark.parametrize('subject_name', ["S0380", "S0563"])
def test_non_regression(setup_prediction, subject_name):
    ref_input_dir, ref_seg_dir = setup_prediction
    label_prefix = MODELS[MODEL_NAME]['multilabel_prefix']
    
    ref_seg_path = next(
        ref_seg_dir.joinpath(subject_name).glob("**/" + label_prefix + "*"))
    predicted_seg_path = next(
        ref_input_dir.joinpath(subject_name).glob("**/" + label_prefix + "*"))
    ref_seg = LoadImage(image_only=True)(ref_seg_path).get_array()
    predicted_seg = LoadImage(image_only=True)(predicted_seg_path).get_array()
    np.testing.assert_almost_equal(ref_seg, predicted_seg)
