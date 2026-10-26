..
  SPDX-FileCopyrightText: 2022 Ludwig Hülk <https://github.com/Ludee> © Reiner Lemoine Institut
  SPDX-FileCopyrightText: super-repo v0.5.0 <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
  SPDX-License-Identifier: MIT

.. figure:: https://user-images.githubusercontent.com/14353512/185425447-85dbcde9-f3a2-4f06-a2db-0dee43af2f5f.png
    :align: left
    :target: https://github.com/OpenEnergyPlatform/energy-technology-catalogs/
    :alt: Repo logo

==========================
energy-technology-catalogs
==========================

**A repo for interoperable data catalogs of energy technologies.**

.. list-table::
   :widths: auto

   * - License
     - |badge_license| |badge_reuse|
   * - Documentation
     - |badge_documentation| |badge_mkdocs|
   * - Tests
     - |badge_tox| |badge_codecov|
   * - Publication
     - |badge_pypi| |badge_python| |badge_pypi_downloads|
   * - Development
     - |badge_issue_open| |badge_issue_closes| |badge_pr_open| |badge_pr_closes|
   * - Community
     - |badge_contributions| |badge_contributors| |badge_repo_counts|

.. contents::
    :depth: 2
    :local:
    :backlinks: top

Introduction
============
| This repository integrates techno-economic data from established energy technology catalogs.
| Parameters such as investment costs, efficiencies, lifetimes and operation and maintenance costs are mapped to a common data structure and described with `OEMetadata <https://github.com/OpenEnergyPlatform/oemetadata>`_ and the `Open Energy Ontology <https://github.com/OpenEnergyPlatform/ontology>`_.
| The goal is to make values from different sources comparable, traceable to their origin and directly usable in energy system modelling and heat planning.

Documentation
=============
| The documentation is created with Markdown using `MkDocs <https://www.mkdocs.org/>`_ and `mike <https://github.com/jimporter/mike>`_.
| All files are stored in the ``docs`` folder of the repository.
| A **GitHub Actions** deploys the ``develop`` branch on a **GitHub Page**.
| The documentation page is: `openenergyplatform.github.io/energy-technology-catalogs/ <https://openenergyplatform.github.io/energy-technology-catalogs/>`_

Collaboration
=============
| Everyone is invited to develop this repository with good intentions.
| Please follow the workflow described in the `CONTRIBUTING.md <https://github.com/OpenEnergyPlatform/energy-technology-catalogs/blob/production/CONTRIBUTING.md>`_.
| AI tools may be used under the conditions of the `AI_COVENANT.md <https://github.com/OpenEnergyPlatform/energy-technology-catalogs/blob/production/AI_COVENANT.md>`_.

Contributors:

.. figure:: https://contrib.rocks/image?repo=OpenEnergyPlatform/energy-technology-catalogs
    :align: left
    :target: https://github.com/OpenEnergyPlatform/energy-technology-catalogs/graphs/contributors
    :alt: [contrib.rocks](https://contrib.rocks)

License and Citation
====================
| The code of this repository is licensed under the **MIT License** (MIT).
| See `LICENSE.txt <https://github.com/OpenEnergyPlatform/energy-technology-catalogs/blob/production/LICENSE.txt>`_ for rights and obligations.
| See the *Cite this repository* function or `CITATION.cff <https://github.com/OpenEnergyPlatform/energy-technology-catalogs/blob/production/CITATION.cff>`_ for citation of this repository.
| Copyright: `energy-technology-catalogs <https://github.com/OpenEnergyPlatform/energy-technology-catalogs/>`_ © `Reiner Lemoine Institut <https://reiner-lemoine-institut.de/>`_ | `MIT <LICENSE.txt>`_


.. |badge_license| image:: https://img.shields.io/github/license/OpenEnergyPlatform/energy-technology-catalogs
    :target: https://github.com/OpenEnergyPlatform/energy-technology-catalogs/blob/production/LICENSE.txt
    :alt: License

.. |badge_reuse| image:: https://api.reuse.software/badge/github.com/OpenEnergyPlatform/energy-technology-catalogs
    :target: https://api.reuse.software/info/github.com/OpenEnergyPlatform/energy-technology-catalogs
    :alt: REUSE

.. |badge_documentation| image:: https://img.shields.io/github/actions/workflow/status/OpenEnergyPlatform/energy-technology-catalogs/documentation.yml?branch=develop&label=documentation
    :target: https://rl-institut.github.io/energy-technology-catalogs/
    :alt: Documentation

.. |badge_mkdocs| image:: https://img.shields.io/badge/Material_for_MkDocs-526CFE?style=flat&logo=MaterialForMkDocs&logoColor=white&color=grey
    :target: https://squidfunk.github.io/mkdocs-material/
    :alt: MkDocs

.. |badge_tox| image:: https://img.shields.io/github/actions/workflow/status/OpenEnergyPlatform/energy-technology-catalogs/tox.yml?label=tox
    :target: https://github.com/OpenEnergyPlatform/energy-technology-catalogs/actions/workflows/tox.yml
    :alt: Tox Tests

.. |badge_codecov| image:: https://codecov.io/gh/OpenEnergyPlatform/energy-technology-catalogs/graph/badge.svg?token=YYCJI3D5G5
    :target: https://codecov.io/gh/OpenEnergyPlatform/energy-technology-catalogs
    :alt: Codecov

.. |badge_pypi| image:: https://img.shields.io/pypi/v/energy-technology-catalogs
    :target: https://pypi.org/project/energy-technology-catalogs/
    :alt: PyPI Version

.. |badge_python| image:: https://img.shields.io/pypi/pyversions/energy-technology-catalogs
    :target: https://github.com/OpenEnergyPlatform/energy-technology-catalogs/blob/develop/pyproject.toml
    :alt: PyPI Python Version

.. |badge_pypi_downloads| image:: https://img.shields.io/pypi/dm/energy-technology-catalogs
    :target: https://pypi.org/project/energy-technology-catalogs/
    :alt: PyPI Downloads

.. |badge_issue_open| image:: https://img.shields.io/github/issues-raw/OpenEnergyPlatform/energy-technology-catalogs
    :target: https://github.com/OpenEnergyPlatform/energy-technology-catalogs/issues
    :alt: Open Issues

.. |badge_issue_closes| image:: https://img.shields.io/github/issues-closed-raw/OpenEnergyPlatform/energy-technology-catalogs
    :target: https://github.com/OpenEnergyPlatform/energy-technology-catalogs/issues?q=is%3Aissue+is%3Aclosed
    :alt: Closed Issues

.. |badge_pr_open| image:: https://img.shields.io/github/issues-pr-raw/OpenEnergyPlatform/energy-technology-catalogs
    :target: https://github.com/OpenEnergyPlatform/energy-technology-catalogs/pulls
    :alt: Open PR

.. |badge_pr_closes| image:: https://img.shields.io/github/issues-pr-closed-raw/OpenEnergyPlatform/energy-technology-catalogs
    :target: https://github.com/OpenEnergyPlatform/energy-technology-catalogs/pulls?q=is%3Apr+is%3Aclosed
    :alt: Closed PR

.. |badge_contributions| image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :target: https://github.com/OpenEnergyPlatform/energy-technology-catalogs/blob/production/CONTRIBUTING.md
    :alt: Contributions

.. |badge_contributors| image:: https://img.shields.io/github/contributors/OpenEnergyPlatform/energy-technology-catalogs
    :target: https://github.com/OpenEnergyPlatform/energy-technology-catalogs/graphs/contributors
    :alt: Contributors

.. |badge_repo_counts| image:: https://hits.sh/github.com/OpenEnergyPlatform/energy-technology-catalogs.svg
    :target: https://hits.sh/github.com/OpenEnergyPlatform/energy-technology-catalogs/
    :alt: Hits
