#
# pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
# Copyright (C) 2020 A. Eugene DePrince III
#
# This file is part of the pdaggerq package.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from typing import Tuple
import copy
import numpy as np

# This imports fake values of all these quantities (zero tensors) so
# we can build the optimal tensor contraction orderings
# NOTE: THESE ARE PROTECTED VARIABLE NAMES FOR THIS MODULE
from pdaggerq.config import (o, v, h, f, g, t1, t2, t3, t4, l1, l2, l3, l4, r1,
                             r2, r3, r4, kd)


class Index:

    def __init__(self, name: str, support: str):
        """
        Generate an index that acts on a particular space

        Later we might  want to generate the axis over which this index
        is used to generate einsum code.  See numpypsi4 for this.

        :param name: how the index shows up
        :param support: where the index ranges. Options: occ,virt,all
        """
        self.name = name
        self.support = support

    def __repr__(self):
        return "{}".format(self.name)

    def __eq__(self, other):
        if not isinstance(other, Index):
            raise TypeError("Can't compare non Index object to Index")
        return other.name == self.name and other.support == self.support

    def __ne__(self, other):
        return not self.__eq__(other)


class BaseTerm:
    """
    Base object for building named abstract tensors

    These objects one can ONLY be composed by multiplication with other BaseTerms
    to produce new TensorTerms.  They can also be checked for equality.
    """

    def __init__(self, *, indices: Tuple[Index, ...], name: str):
        self.name = name
        self.indices = indices

    def __repr__(self):
        return "{}".format(self.name) + "(" + ",".join(
            repr(xx) for xx in self.indices) + ")"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __mul__(self, other):
        # what about numpy floats  and such?
        if isinstance(other, BaseTerm):
            return TensorTerm((copy.deepcopy(self), other))
        elif isinstance(other, TensorTerm):
            self_copy = copy.deepcopy(self)
            return other.__mul__(self_copy)
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        name_equality = other.name == self.name
        if len(self.indices) == len(other.indices):
            index_equal = all([self.indices[xx] == other.indices[xx] for xx in
                               range(len(self.indices))])
            return name_equality and index_equal
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class TensorTermAction:
    """
    Object for representing a type of transform on TensorTerm.

    An example of Transformation can be duplicate or permute indices,

    permute action will be used in conjunction with einsum contraction to
    minimize contraction work.
    """

    def __init__(self, *, indices: Tuple[Index, ...], name: str):
        self.name = name
        self.indices = indices

    def __repr__(self):
        return "{}".format(self.name) + "(" + ",".join(
            repr(xx) for xx in self.indices) + ")"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        name_equality = other.name == self.name
        if len(self.indices) == len(other.indices):
            index_equal = all([self.indices[xx] == other.indices[xx] for xx in
                               range(len(self.indices))])
            return name_equality and index_equal
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class TensorTerm:
    """
    collection  of BaseTerms that can be translated to a einsnum contraction
    """

    def __init__(self, base_terms: Tuple[BaseTerm, ...], coefficient=1.0,
                 permutation_ops=None):
        self.base_terms = base_terms
        self.coefficient = coefficient
        if permutation_ops is not None:
            if len(permutation_ops) == 0:
                self.actions = None
            else:
                self.actions = permutation_ops
        else:
            self.actions = None

    def __repr__(self):
        if self.actions is None:
            return "{: 5.4f} ".format(self.coefficient) + "*".join(
                xx.__repr__() for xx in self.base_terms)
        else:
            return "{: 5.4f} ".format(self.coefficient) + "*".join(
                xx.__repr__() for xx in self.actions) + "*".join(
                xx.__repr__() for xx in self.base_terms)

    def __mul__(self, other):
        self_copy = copy.deepcopy(self)
        if isinstance(other, (int, float, complex)):
            self_copy.coefficient *= other
        elif isinstance(other, BaseTerm):
            self_copy.base_terms = self_copy.base_terms + (other,)
        return self_copy

    def __rmul__(self, other):
        return self.__mul__(other)

    def einsum_string(self, update_val,
                      output_variables=None,
                      occupied=['i', 'j', 'k', 'l', 'm', 'n'],
                      virtual=['a', 'b', 'c', 'd', 'e', 'f'],
                      occ_char=None,
                      virt_char=None,
                      optimize=True):
        einsum_out_strings = ""
        einsum_tensors = []
        tensor_out_idx = []
        einsum_strings = []
        if occ_char is None:
            # in our code this will be a slice. o = slice(None, nocc)
            occ_char = 'o'
        if virt_char is None:
            virt_char = 'v'  # v = slice(nocc, None)

        for bt in self.base_terms:
            tensor_index_ranges = []
            string_indices = [xx.name for xx in bt.indices]
            for idx_type in string_indices:
                if idx_type in occupied:
                    if bt.name in ['h', 'g', 'f', 'kd']:
                        tensor_index_ranges.append(occ_char)
                    else:
                        tensor_index_ranges.append(':')
                    if output_variables is not None:
                        if idx_type in output_variables:
                            tensor_out_idx.append(idx_type)
                elif idx_type in virtual:  # virtual
                    if bt.name in ['h', 'g', 'f', 'kd']:
                        tensor_index_ranges.append(virt_char)
                    else:
                        tensor_index_ranges.append(':')
                    if output_variables is not None:
                        if idx_type in output_variables:
                            tensor_out_idx.append(idx_type)
                else:  # route to output with ->
                    tensor_index_ranges.append(idx_type)

            # HACK for adcc
            if bt.name in ['t1', 't2', 't3', 'l2', 'l1', 'r1', 'r2']:
                einsum_tensors.append(bt.name)
                if len(string_indices) == 2:
                    string_indices = string_indices[::-1]
                elif len(string_indices) == 4:
                    si = string_indices
                    string_indices = [si[2], si[3], si[0], si[1]]
            # HACK for adcc
            elif bt.name == 'f':
                einsum_tensors.append('hf.f' + "".join(tensor_index_ranges))
            elif bt.name == 'g':
                def to_canonical(blks):
                    blk = "".join(blks)
                    canblks = ['oooo', 'ooov', 'oovv', 'ovov', 'ovvv', 'vvvv']
                    if blk in canblks:
                        return "", (0, 1, 2, 3), blk
                    ps = [
                        (-1, (1, 0, 2, 3)),
                        (-1, (0, 1, 3, 2)),
                        (+1, (1, 0, 3, 2)),
                        (+1, (2, 3, 0, 1)),
                        (-1, (3, 2, 0, 1)),
                    ]
                    for p in ps:
                        lbl = "".join([blks[i] for i in p[1]])
                        if lbl in canblks:
                            sgn = "-" if p[0] == -1 else ""
                            return sgn, p[1], lbl
                    raise ValueError(blks)
                sgn, perm, blk = to_canonical(tensor_index_ranges)
                string_indices = [string_indices[i] for i in perm]
                einsum_tensors.append(f'{sgn}hf.' + blk)
            else:
                einsum_tensors.append(
                    bt.name + "[" + ", ".join(tensor_index_ranges) + "]")
            einsum_strings.append("".join(string_indices))
        if tensor_out_idx:
            out_tensor_ordered = list(filter(None, [
                xx if xx in tensor_out_idx else None for xx in
                output_variables]))
            einsum_out_strings += "->{}".format("".join(out_tensor_ordered))

        teinsum_string = "= {: 5.15f} * einsum(\'".format(self.coefficient)

        if len(einsum_strings) > 2 and optimize:
            einsum_path_string = "np.einsum_path(\'".format(self.coefficient)
            einsum_path_string += ",".join(
                einsum_strings) + einsum_out_strings + "\', " + ", ".join(
                einsum_tensors) + ", optimize=\'optimal\')"
            einsum_optimal_path = eval(einsum_path_string)
            # print(einsum_optimal_path[1])
            teinsum_string += ",".join(
                einsum_strings) + einsum_out_strings + "\', " + ", ".join(
                einsum_tensors) + ", optimize={})".format(
                einsum_optimal_path[0])
        else:
            teinsum_string += ",".join(
                einsum_strings) + einsum_out_strings + "\', " + ", ".join(
                einsum_tensors) + ")"
        if update_val is not None and self.actions is None:
            teinsum_string = update_val + " " + '+' + teinsum_string
        elif update_val is not None and self.actions is not None:
            # now generate logic for single permutation
            original_out = list(filter(None,
                                       [xx if xx in tensor_out_idx else None for
                                        xx in output_variables]))
            outstrings = [[+1, original_out]]

            antisymm = []
            for act in self.actions:
                # check if we have a permutation
                if not isinstance(act, ContractionPermuter):
                    raise NotImplementedError(
                        "currently only permutations are implemented")
                # print(act)
                # get all sets of exchanged indices
                exchanged_indices = [xx.name for xx in act.indices]
                # print(exchanged_indices, outstrings)

                permuted_outstrings = []
                for perm in outstrings:
                    # Generate new permuted outstring by
                    # a) make copy of current permutation
                    # b) find position of indices in string
                    # c) swap positions of indices
                    tmp_outsrings = copy.deepcopy(perm[1])
                    ii, jj = tmp_outsrings.index(
                        exchanged_indices[0]), tmp_outsrings.index(
                        exchanged_indices[1])
                    # print(ii, jj)
                    tmp_outsrings[ii], tmp_outsrings[jj] = tmp_outsrings[jj], \
                                                           tmp_outsrings[ii]
                    antisymm.append((ii, jj))
                    # store permuted set with a -1 phase
                    permuted_outstrings.append([perm[0] * -1, tmp_outsrings])

                # outstrings = outstrings + permuted_outstrings

            # now that we've generated all the permutations
            # generate the reshapped summands.
            # HACK for adcc
            teinsum_string = update_val + " " + '+' + teinsum_string
            for a in antisymm:
                teinsum_string += f".antisymmetrise({a[0]}, {a[1]})"
            teinsum_string += f"*{2**len(antisymm)}"
            
            # teinsum_string = 'contracted_intermediate' + " " + teinsum_string + ".evaluate()" + "\n"
            # teinsum_string += update_val
            # teinsum_string += " += "
            # update_val_line = []
            # for ots in outstrings:
            #     new_string = ""
            #     new_string += '{: 5.5f} * '.format(ots[0])
            #     if tuple(ots[1]) == tuple(original_out):
            #         new_string += 'contracted_intermediate'
            #     else:
            #         new_string += 'einsum(\'{}->{}\', contracted_intermediate) '.format(
            #             ''.join(original_out), "".join(ots[1]))
            #     update_val_line.append(new_string)
            # teinsum_string += " + ".join(update_val_line)
        return teinsum_string


class Right0amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='r0'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "r0"


class Right1amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='r1'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "r1({},{})".format(self.indices[0], self.indices[1])


class Right2amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='r2'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "r2({},{},{},{})".format(self.indices[0], self.indices[1],
                                        self.indices[2], self.indices[3])

class Right3amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='r3'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "r3({},{},{},{},{},{})".format(self.indices[0], self.indices[1],
                                              self.indices[2], self.indices[3],
                                              self.indices[4], self.indices[5])


class Right4amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='r4'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "r4({},{},{},{},{},{},{},{})".format(self.indices[0],
                                                    self.indices[1],
                                                    self.indices[2],
                                                    self.indices[3],
                                                    self.indices[4],
                                                    self.indices[5],
                                                    self.indices[6],
                                                    self.indices[7])
class Left0amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='l0'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "l0"


class Left1amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='l1'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "l1({},{})".format(self.indices[0], self.indices[1])


class Left2amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='l2'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "l2({},{},{},{})".format(self.indices[0], self.indices[1],
                                        self.indices[2], self.indices[3])


class Left3amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='l3'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "l3({},{},{},{},{},{})".format(self.indices[0], self.indices[1],
                                              self.indices[2], self.indices[3],
                                              self.indices[4], self.indices[5])


class Left4amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='l4'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "l4({},{},{},{},{},{},{},{})".format(self.indices[0],
                                                    self.indices[1],
                                                    self.indices[2],
                                                    self.indices[3],
                                                    self.indices[4],
                                                    self.indices[5],
                                                    self.indices[6],
                                                    self.indices[7]
                                                    )


class D1(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='d1'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "d1({},{})".format(self.indices[0], self.indices[1])


class T1amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='t1'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "t1({},{})".format(self.indices[0], self.indices[1])


class T2amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='t2'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "t2({},{},{},{})".format(self.indices[0], self.indices[1],
                                        self.indices[2], self.indices[3])

class T3amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='t3'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "t3({},{},{},{},{},{})".format(self.indices[0], self.indices[1],
                                              self.indices[2], self.indices[3],
                                              self.indices[4], self.indices[5])


class T4amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='t4'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "t4({},{},{},{},{},{},{},{})".format(self.indices[0],
                                                    self.indices[1],
                                                    self.indices[2],
                                                    self.indices[3],
                                                    self.indices[4],
                                                    self.indices[5],
                                                    self.indices[6],
                                                    self.indices[7])



class OneBody(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='h'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "h({},{})".format(self.indices[0], self.indices[1])


class FockMat(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='f'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "f({},{})".format(self.indices[0], self.indices[1])


class TwoBody(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='g'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "<{},{}||{},{}>".format(self.indices[0], self.indices[1],
                                       self.indices[2], self.indices[3])


class Delta(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='kd'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "d({},{})".format(self.indices[0], self.indices[1])


class ContractionPermuter(TensorTermAction):

    def __init__(self, *, indices=Tuple[Index, ...], name='P'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "P({},{})".format(self.indices[0], self.indices[1])
