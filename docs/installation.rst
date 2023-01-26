Installation
============

Simply run to install the package:

.. code-block:: bash
    
    pip install aac-metrics

Then download the external tools needed for SPICE, PTBTokenizer, METEOR and FENSE:

.. code-block:: bash
    
    aac-metrics-download


Python requirements
###################

The python requirements are automatically installed when using pip on this repository.

.. code-block:: bash

    torch>=1.10.1
    numpy>=1.21.2
    pyyaml>=6.0
    tqdm>=4.64.0
    sentence-transformers>=2.2.2
