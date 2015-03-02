

class CIF(object):
    """Reads and writes .cif files. Adheres to the IUCR conventions for 
    keywords and directives. For details see (valid as of Feb. 2015)
    http://www.iucr.org/__data/iucr/cifdic_html/1/cif_core.dic/index.html
    """

    def __init__(self):
        self._filename = None



