"""
CIF format file I/O operations.
"""
import shlex
from datetime import date


class CIF(object):

    def __init__(self, name="structure", file=None):
        self.name = name
        self._data = {}
        self._headings = {}
        self._element_labels = {}
        self.non_loops = ["data", "cell", "sym", "end"]
        self.block_order = ["data", "sym", "sym_loop", "cell", "atoms", "bonds"]
        if file is not None:
            self.read(file)

    def read(self, filename):
        filestream = open(filename, 'r')
        filelines = filestream.readlines()
        blocks = []
        loopcount = 0
        loopentries = {}
        loopread = False
        blockread = False
        self.block_order = []

        for line in filelines:
            #line=line.replace("\n", "")
            # why not strip here?
            line = line.strip()
            if line.startswith("data_"):
                self.name = line[5:]
                self.insert_block_order("data")
                self.add_data("data", data_=self.name)

            if loopread and line.startswith("_"):
                # change block names, easier for adding data to structure graph
                if ('_atom_' in line and '_bond_' not in line):
                    self.insert_block_order('atoms', loopcount, _REPLACE=True)
                elif ('_geom_bond' in line):
                    self.insert_block_order('bonds', loopcount, _REPLACE=True)
                loopentries[loopcount].append(line)



            elif loopread and not line.startswith("_"):
                loopread = False
                blockread = True

            elif not loopread and line.startswith("_"):
                block = self.get_non_loop_block(line)
                self.insert_block_order(block)
                # hopefully all non-loop entries are just single value entries,
                # otherwise this is invalid.
                try:
                    key, val = line.strip().split()
                except ValueError:
                    key, val = line.strip().split()[:2]
                if val.endswith("(0)"):
                    val = val[:-3]
                self.add_data(block, **{key.strip():self.general_label(val)})

            if blockread and (line.startswith("loop_") or line.startswith("_") or not line):
                blockread = False

            if line == "loop_":
                loopcount += 1
                loopentries[loopcount] = []
                loopread = True
                blockread = False
                self.insert_block_order(loopcount)

            if blockread:
                #split_line = line.strip().split()
                split_line = shlex.split(line)
                assert len(loopentries[loopcount]) == len(split_line)
                for key, val in zip(loopentries[loopcount], split_line):
                    self.add_data(loopcount, **{key:self.general_label(val)})

        filestream.close()

    def get_time(self):
        t = date.today()
        return t.strftime("%A %d %B %Y")

    def insert_block_order(self, name, index=None, _REPLACE=False):
        """Adds a block to the cif file in a specified order, unless index is specified,
        will not override existing order"""
        if index is None and name in self.block_order:
            return
        elif index is not None and (self.block_order[index] == name):
            return
        elif index is None and name not in self.block_order:
            index = len(self.block_order)
        elif index is not None and name in self.block_order and index < len(self.block_order) and not _REPLACE:
            old = self.block_order.index(name)
            self.block_order.pop(old)
        elif index is not None and name not in self.block_order and index < len(self.block_order) and _REPLACE:
            self.block_order.pop(index)
        elif index is not None and name in self.block_order and index >= len(self.block_order):
            old = self.block_order.index(name)
            self.block_order.pop(old)
            index = len(self.block_order)
        self.block_order = self.block_order[:index] + [name] + \
                            self.block_order[index:]

    def add_data(self, block, **kwargs):
        self._headings.setdefault(block, [])
        for key, val in kwargs.items():
            try:
                self._data[key].append(val)
            except KeyError:
                self._headings[block].append(key)
                if block in self.non_loops:
                    self._data[key] = val
                else:
                    self._data[key] = [val]
            except:
                print(self._data.keys())

    def get_element_label(self, el):
        self._element_labels.setdefault(el, 0)
        self._element_labels[el] += 1
        return el + str(self._element_labels[el])

    def __str__(self):
        line = ""
        for block in self.block_order:
            # NOTE still be able to write CIFS if blond bock not specified
            if block in self._headings:
                heads = self._headings[block]
                if block in self.non_loops:
                    vals = zip([CIF.label(i) for i in heads], [self._data[i] for i in heads])
                else:
                    line += "loop_\n"+"\n".join([CIF.label(i) for i in heads])+"\n"
                    vals = zip(*[self._data[i] for i in heads])
                for ll in vals:
                    line += "".join(ll) + "\n"
        return line

    def get_non_loop_block(self, line):
        if line.startswith("_cell"):
            return "cell"
        elif line.startswith("_symmetry"):
            return "sym"
        elif line.startswith("_audit"):
            return "data"

    # terrible idea for formatting.. but oh well :)
    @staticmethod
    def atom_site_fract_x(x):
        return "%10.5f "%(x)
    @staticmethod
    def atom_site_fract_y(x):
        return "%10.5f "%(x)
    @staticmethod
    def atom_site_fract_z(x):
        return "%10.5f "%(x)
    @staticmethod
    def atom_type_partial_charge(x):
        return "%10.5f "%(x)
    @staticmethod
    def atom_site_label(x):
        return "%-7s "%(x)
    @staticmethod
    def atom_site_type_symbol(x):
        return "%-6s "%(x)
    @staticmethod
    def atom_site_description(x):
        return "%-5s "%(x)
    @staticmethod
    def geom_bond_atom_site_label_1(x):
        return "%-7s "%(x)
    @staticmethod
    def geom_bond_atom_site_label_2(x):
        return "%-7s "%(x)
    @staticmethod
    def geom_bond_distance(x):
        return "%7.3f "%(x)
    @staticmethod
    def geom_bond_site_symmetry_2(x):
        return "%-5s "%(x)
    @staticmethod
    def ccdc_geom_bond_type(x):
        return "%5s "%(x)
    @staticmethod
    def cell_length_a(x):
        return "%-7.4f "%(x)
    @staticmethod
    def cell_length_b(x):
        return "%-7.4f "%(x)
    @staticmethod
    def cell_length_c(x):
        return "%-7.4f "%(x)
    @staticmethod
    def cell_angle_alpha(x):
        return "%-7.4f "%(x)
    @staticmethod
    def cell_angle_beta(x):
        return "%-7.4f "%(x)
    @staticmethod
    def cell_angle_gamma(x):
        return "%-7.4f "%(x)
    @staticmethod
    def atom_site_fragment(x):
        return "%-4i "%(x)
    @staticmethod
    def atom_site_constraints(x):
        return "%-4i "%(x)

    @staticmethod
    def label(x):
        """special cases"""
        if x == "data_":
            return x
        elif x == "_symmetry_space_group_name_H_M":
            # replace H_M with H-M.
            x = x[:28] + "-" + x[29:]
        return "%-34s"%(x)

    @staticmethod
    def general_label(x):
        return "%s     "%(x)
def get_time():
    t = date.today()
    return t.strftime("%A %d %B %Y")
