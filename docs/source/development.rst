Development
===========

Build the documentation locally
-------------------------------

Build the documentation using::

    pip install -e .[docs]  # install docs extra
    cd docs/
    make html  # build html documentation
    
After building, find the documentation in ``docs/build/html/index.html``.

Sphinx cheat sheet
------------------

  * Add code with syntax highlighting like this:

    .. code:: python

        def f(var):
            return "string"

  * Write your mathematical formulae using LaTeX,
    in line :math:`\exp(-i2\pi)` or displayed

    .. math:: f(x) = \int_0^\infty \exp\left(\frac{x^2}{2}\right)\,dx

  * You want to refer to a particular function or class? You can!

    .. autoclass:: lammps_interface.structure_data.MolecularGraph
       :noindex:

  * Check out the source of any page via the link
    in the bottom right corner.

|

reST source of this page:

.. literalinclude:: development.rst

