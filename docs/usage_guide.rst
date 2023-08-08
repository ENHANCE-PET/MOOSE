MOOSE 2.0: How to Harness Its Power
===================================

Command-Line Mastery
---------------------

Engage with MOOSE 2.0's command-line tool effortlessly. The tool requires the directory containing your images and the desired segmentation model.

.. code-block:: bash

   moosez -d <path_to_image_dir> -m <model_name>

For guidance:

.. code-block:: bash

   moosez -h

Library Integration
-------------------

Incorporate MOOSE 2.0 into your Python projects:

.. code-block:: python

   from moosez import moose
   moose(model_name, input_dir, output_dir, accelerator)
