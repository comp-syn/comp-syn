comp-syn
~~~~~~~~

comp-syn is a python package which provides a novel methodology to explore relationships between abstract concepts and color. We provide functionalities to create a dataset for word - image associations through Google Cloud and WordNet, as and demonstrate a variety of image and word related analyses. We introduce constructs such as *Colorgrams* as a tool for analysis, as well as the ability to create *color representational vectors* for images, and words as aggregations of images. Our work demonstrates a strong relationship between abstract semantic domains and colors, and supports claims for collective cognition and embodied cognition. The package allows you to explore these relationships yourself!

The theoretical frameword is described in these papers:

Bhargav Srinivasa Desikan, Tasker Hull, Ethan O Nadler, Douglas Guilbeault, Aabir Abubaker Kar, Mark Chu, Donald Ruggiero Lo Sardo. **comp-syn: Perceptually Grounded Word Embeddings with Color**, *Proceedings of the 28th International Conference on Computational Linguistics (COLING'20)* `link to paper <https://arxiv.org/abs/2010.04292>`__

Douglas Guilbeault , Ethan Nadler, Mark Chu, Ruggiero Lo Sardo, Aabir Abubaker Kar, and Bhargav Srinivasa Desikan. **Color Associations in Abstract Semantic Domains.** *Cognition* (2020). `link to paper <https://www.sciencedirect.com/science/article/abs/pii/S0010027720301256>`__

The work arose through a collaboration between the contributors at the Santa Fe Institute's Complex System Summer School 2019. 


Documentation and Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~

The
`notebooks <https://github.com/comp-syn/comp-syn/tree/master/notebooks>`__
directory showcases the usage of compsyn, with examples and basic usage.

We currently provide functionality to download and organise image data to be loaded as vector objects, as well as a preliminary analysis using them.


Installation
~~~~~~~~~~~~

In a virtual environment:

Run ``pip install compsyn`` to download and install from PyPI.

Run ``pip install -r requirements.txt`` to install from the last-pinned dependencies.

Run ``python setup.py install`` for default installation.

Run ``pip install .`` to install from source.

With `poetry <https://python-poetry.org/>_`:

Run ``poetry install``


Requirements
~~~~~~~~~~~~

-  Python 3.4+
-  ``jzazbz_array.npy`` available here: `link <https://drive.google.com/file/d/1wspjIBzzvO-ZQbiQs3jgN4UETMxTVD2c/view>`_ 
-  A webdriver, like `geckodriver <https://github.com/mozilla/geckodriver/releases>`_ or `chromedriver <https://chromedriver.chromium.org/>`_.


Configuration
~~~~~~~~~~~~~

``compsyn`` requires some configuration be set, more details are available in the configuration notebook.

Notes
~~~~~

To use the package one would need to download the JzAzBz array to convert the images to the appropriate vector form. It can be downloaded here (`link <https://drive.google.com/file/d/1wspjIBzzvO-ZQbiQs3jgN4UETMxTVD2c/view>`_). We provide sample images downloaded from Google Images to test the functionality.



Contributing
~~~~~~~~~~~~

Contributions to compsyn should use docstrings to document each package/class/method being contributed. Tests should also be implemented for as much of the code as possible.


Tests with ``pytest`` - 

To run pytest, first install it (``pip install pytest pytest-cov pytest-depends``). By default, pytest will run every method found in the ``tests`` directory that has the name prefix ``test_``. Tests should also be marked with one of the following decorators: 

  - ``@pytest.mark.unit`` for testing the smallest units of functionality. These tests should be fast, and involve as few dependencies as possible to test each specific method being implemented.
  - ``@pytest.mark.integration`` for testing more abstracted interfaces into compsyn, these tests will involve multiple compsyn code paths being executed as part of some higher order logic. These tests should still be fast, and should not involve external networking.
  - ``@pytest.mark.online`` for testing parts of compsyn that involve network requests. These tests can be arbitrarily complex, and may be slow to run.

by default, all test types will be run by ``pytest``

``pytest --cov=compsyn``

Specific test can be targetted by path and mark,

``pytest -m unit ./tests/test_utils.py``

Compsyn V1.0.0
~~~~~~~~~~~~~~

- New image analysis techniques (wavelet with PCA)
- Introduces the Trial dataclass for creating logically separated datasets
- Optional shared backend via S3, trials can be saved with specific "revision" names, data easily shared
- Re-worked notebooks as documentation of the API to Vectors


Shared Backend with S3
~~~~~~~~~~~~~~~~~~~~~~

- raw data and Vector objects can be saved remotely for collaboration
- saving vectors or data to the backend is done by creating a "revision"

For using pre-computed JzAzBz and RGB color distributions for words, please download (`link for vectors <https://drive.google.com/file/d/13J3QHn4NPdCTEkTctVYFqfYYJEMAOgAT/view?usp=sharing>`_) and (`link for concreteness values <https://drive.google.com/file/d/1edQaibCW9yCih_pVeeYZWzPlgNZJ4Fzp/view?usp=sharing>`_).

