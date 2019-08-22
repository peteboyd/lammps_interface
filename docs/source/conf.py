# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'lammps-interface'
copyright = '2019, Peter Boyd, Mohamad Moosavi, Matthew Witman'
author = 'Peter Boyd, Mohamad Moosavi, Matthew Witman'

# The full version, including alpha/beta/rc tags
release = '0.1.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for ReadTheDocs -------------------------------------------------

def run_apidoc(_):
    """Runs sphinx-apidoc when building the documentation.

    Needs to be done in conf.py in order to include the APIdoc in the
    build on readthedocs.

    See also https://github.com/rtfd/readthedocs.org/issues/1139
    """
    import subprocess

    source_dir = os.path.abspath(os.path.dirname(__file__))
    apidoc_dir = os.path.join(source_dir, 'apidoc')
    package_dir = os.path.join(source_dir, os.pardir, os.pardir, 'lammps_interface')

    # In #1139, they suggest the route below, but this ended up
    # calling sphinx-build, not sphinx-apidoc
    #from sphinx.apidoc import main
    #main([None, '-e', '-o', apidoc_dir, package_dir, '--force'])

    cmd_path = 'sphinx-apidoc'
    if hasattr(sys, 'real_prefix'):  # Check to see if we are in a virtualenv
        # If we are, assemble the path manually
        cmd_path = os.path.abspath(
            os.path.join(sys.prefix, 'bin', 'sphinx-apidoc'))

    options = [
        '-o',
        apidoc_dir,
        package_dir,
        '--private',
        '--force',
        '--no-toc',
    ]

    # See https://stackoverflow.com/a/30144019
    env = os.environ.copy()
    env["SPHINX_APIDOC_OPTIONS"] = 'members,special-members,private-members,undoc-members,show-inheritance'
    subprocess.check_call([cmd_path] + options, env=env)


def setup(app):
    app.connect('builder-inited', run_apidoc)
