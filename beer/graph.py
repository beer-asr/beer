'Acoustic graph for the HMM.'

from collections import defaultdict, OrderedDict
import torch
from .utils import logsumexp

class Arc:
    '''Arc between to state (i.e. node) of a graph with a weight.

    Attributes:
        start (int): Identifier of the starting state.
        end (int): Identifier of the ending state.
        weight (float): Weight of the arc.
    '''
    __repr_str = 'Arc(start={}, end={}, weight={})'

    def __init__(self, start, end, weight):
        '''
        Args:
            start (State): Starting state.
            end (State): Final state,
            weight (int): Weight of the arc.
        '''
        self.start = start
        self.end = end
        self.weight = weight

    def __hash__(self):
        return hash('{}:{}'.format(self.start, self.end))

    def __repr__(self):
        return self.__repr_str.format(self.start, self.end, self.weight)


class State:
    '''State (or node) of a graph.

    Attributes:
        unit_id (obj): Identifier of the parent unit.
        state_id (int): Unique identifier of the state within the unit.
        pdf_id (int): Idenitifier of the probability density function
            associated with this state.
    '''
    __repr_str = 'State(state_id={}, pdf_id={})'

    def __init__(self, state_id, pdf_id):
        self.state_id = state_id
        self.pdf_id = pdf_id

    def __hash__(self):
        return hash(self.state_id)

    def __repr__(self):
        return self.__repr_str.format(self.state_id, self.pdf_id)

    def __eq__(self, other):
        if isinstance(other, State):
            return hash(self) == hash(other)
        return NotImplemented


def _state_name(symbols, state_id):
    try:
        name = str(symbols[state_id])
    except KeyError:
        name = str(state_id)
    return name

class Graph:
    '''Graph.

    Attributes:
        states (dictionary): All the states of the graph.
        arcs (dictionary): All the arcs of the graph.
    '''

    def __init__(self):
        self._state_count = 0
        self._states = OrderedDict()
        self._arcs = set()
        self.symbols = {}
        self.start_state = None
        self.end_state = None

    def __repr__(self):
        retval = ''
        for i, state in enumerate(self._states):
            retval += repr(state)
            if i <= len(self._states) - 1:
                retval += '\n'
        return retval

    def _repr_svg_(self):
        # We import the module here as it is only needed by the Jupyter
        # notebook.
        import graphviz
        dot = graphviz.Digraph()
        dot.graph_attr['rankdir'] = 'LR'
        for state_id in self.states():
            attrs = {'shape': 'circle'}
            if state_id == self.start_state:
                attrs.update(penwidth='2.0')
            if state_id == self.end_state:
                attrs.update(shape='doublecircle')
            dot.node(_state_name(self.symbols, state_id), **attrs)
        for arc in self.arcs():
            dot.edge(_state_name(self.symbols, arc.start),
                     _state_name(self.symbols, arc.end),
                     label=str(round(arc.weight, 3)))
        return graphviz.Source(dot.source)._repr_svg_()

    def states(self):
        '''Iterate over the states.'''
        return self._states.keys()

    def arcs(self, state_id=None, incoming=False):
        '''Iterates over the arcs. If state is provided enumerate the
        outgoing args from "state_id"
        '''
        for arc in self._arcs:
            if not incoming:
                if arc.start == state_id or state_id is None:
                    yield arc
            else:
                if arc.end == state_id or state_id is None:
                    yield arc

    def add_state(self, pdf_id=None):
        state_id = self._state_count
        self._state_count += 1
        new_state = State(state_id, pdf_id)
        self._states[new_state.state_id] = new_state
        return state_id

    def add_arc(self, start, end, weight=1.0):
        new_arc = Arc(start, end, weight)
        self._arcs.add(new_arc)
        return new_arc

    def normalize(self):
        for state_id in self.states():
            sum_out_weights = 0.
            for arc in self.arcs(state_id):
                sum_out_weights += arc.weight
            for arc in self.arcs(state_id):
                arc.weight /= sum_out_weights

    def replace_state(self, old_state_id, graph):
        '''Replace a state with a graph.'''

        # Copy the states.
        new_states = {}
        for state_id in graph.states():
            pdf_id = graph._states[state_id].pdf_id
            new_state_id = self.add_state(pdf_id=pdf_id)
            new_states[state_id] = new_state_id

        # Copy the arcs.
        for arc in graph.arcs():
            self.add_arc(new_states[arc.start], new_states[arc.end], arc.weight)

        # Connect the unit graph to the main graph.
        to_delete = []
        new_arcs = []
        for arc in self.arcs(old_state_id):
            to_delete.append(arc)
            new_arcs.append((new_states[graph.end_state], arc.end, arc.weight))
        for arc in self.arcs(old_state_id, incoming=True):
            to_delete.append(arc)
            new_arcs.append((arc.start, new_states[graph.start_state], arc.weight))

        # Add the new arcs.
        for start, end, weight in new_arcs:
            self.add_arc(start, end, weight)

        # Remove the old arcs and the replaced state.
        for arc in to_delete:
            self._arcs.remove(arc)
        del self._states[old_state_id]

    def _find_next_pdf_ids(self, start_state, init_weight):
        to_explore = [arc for arc in self.arcs(start_state)]
        visited = set([start_state])
        while len(to_explore) > 0:
            arc = to_explore.pop()
            pdf_id = self._states[arc.end].pdf_id
            if pdf_id is not None:
                yield arc.end, init_weight * arc.weight
            else:
                if arc.end not in visited:
                    to_explore += [arc for arc in self.arcs(arc.end)]
                    visited.add(arc.end)

    def _find_previous_pdf_ids(self, start_state, init_weight):
        to_explore = [arc for arc in self.arcs(start_state, incoming=True)]
        visited = set([start_state])
        while len(to_explore) > 0:
            arc = to_explore.pop()
            pdf_id = self._states[arc.start].pdf_id
            if pdf_id is not None:
                yield arc.start, init_weight * arc.weight
            else:
                if arc.start not in visited:
                    to_explore += [arc for arc in self.arcs(arc.start, incoming=True)]
                    visited.add(arc.start)

    def compile(self):
        '''Compile the graph.'''
        # Total number of emitting states.
        tot_n_states = 0
        pdf_id_mapping = []
        state2pdf_id = {}
        for state_id, state in self._states.items():
            if state.pdf_id is not None:
                state2pdf_id[state_id] = tot_n_states
                pdf_id_mapping.append(state.pdf_id)
                tot_n_states += 1

        init_probs = torch.zeros(tot_n_states)
        final_probs = torch.zeros(tot_n_states)
        trans_probs = torch.zeros(tot_n_states, tot_n_states)

        # Init probs.
        for state_id, weight in self._find_next_pdf_ids(self.start_state, 1.0):
            init_probs[state2pdf_id[state_id]] += weight
        init_probs /= init_probs.sum()

        # Init probs.
        for state_id, weight in self._find_previous_pdf_ids(self.end_state, 1.0):
            final_probs[state2pdf_id[state_id]] += weight
        final_probs /= final_probs.sum()

        # Transprobs
        for arc in self.arcs():
            pdf_id1 = self._states[arc.start].pdf_id
            pdf_id2 = self._states[arc.end].pdf_id
            weight = arc.weight

            # These connections are handled by the init_probs.
            if pdf_id1 is None:
                continue

            # We need to follow the path until the next valid pdf_id
            pdf_id1 = state2pdf_id[arc.start]
            if pdf_id2 is None:
                for state_id, weight in self._find_next_pdf_ids(arc.end, weight):
                    trans_probs[pdf_id1, state2pdf_id[state_id]] += weight
            else:
                trans_probs[pdf_id1, state2pdf_id[arc.end]] += weight

        # Normalize the transition matrix withouth changing its diagonal.
        for dim in range(len(trans_probs)):
            diag = trans_probs[dim, dim].clone()
            off_diag  = trans_probs[dim, :].sum() - diag
            if diag > 0. and off_diag > 0:
                norms = off_diag
                trans_probs[dim, :] /= norms / (1 - diag)
                trans_probs[dim, dim] =  diag

        return CompiledGraph(init_probs, final_probs, trans_probs, pdf_id_mapping)


class CompiledGraph:
    '''Inference graph for a HMM model.'''

    def __init__(self, init_probs, final_probs, trans_probs, pdf_id_mapping=None):
        '''
        Args:
            init_probs (``torch.Tensor``): Initial probabilities.
            final_probs (``torch.Tensor``): Final probabilities.
            trans_probs (``torch.Tensor``): Transition probabilities.
            pdf_id_mapping (list): Mapping of the pdf ids (optional)
        '''
        self.init_probs = init_probs
        self.final_probs = final_probs
        self.trans_probs = trans_probs
        self.pdf_id_mapping = pdf_id_mapping

    @property
    def n_states(self):
        'Total number of states in the graph.'
        return len(self.trans_probs)

    def _baum_welch_forward(self, lhs):
        alphas = torch.zeros_like(lhs)
        consts = torch.zeros(len(lhs), dtype=lhs.dtype, device=lhs.device)
        trans_mat = self.trans_probs
        res = lhs[0] * self.init_probs
        consts[0] = res.sum()
        alphas[0] = res / consts[0]
        for i in range(1, lhs.shape[0]):
            res = lhs[i] * (trans_mat.t() @ alphas[i-1])
            consts[i] = res.sum()
            alphas[i] = res / consts[i]
        return alphas, consts

    def _baum_welch_backward(self, lhs, consts):
        betas = torch.zeros_like(lhs)
        trans_mat = self.trans_probs
        betas[-1] = self.final_probs
        for i in reversed(range(lhs.shape[0] - 1)):
            res = trans_mat @ (lhs[i+1] * betas[i+1])
            betas[i] = res / consts[i+1]
        return betas

    def posteriors(self, llhs):
        # Scale the log-likelihoods to avoid overflow.
        max_val = llhs.max()
        lhs = (llhs - max_val).exp() + 1e-9

        # Scaled forward-backward algorithm.
        alphas, consts = self._baum_welch_forward(lhs)
        betas = self._baum_welch_backward(lhs, consts)
        posts = alphas * betas * torch.exp(max_val)
        norm = posts.sum(dim=1)
        posts /= norm[:, None]

        return posts

    def best_path(self, llhs):
        init_log_prob = self.init_probs.log()
        backtrack = torch.zeros_like(llhs, dtype=torch.long, device=llhs.device)
        omega = llhs[0] + init_log_prob
        log_trans_mat = self.trans_probs.log()

        for i in range(1, llhs.shape[0]):
            hypothesis = omega + log_trans_mat.t()
            backtrack[i] = torch.argmax(hypothesis, dim=1)
            omega = llhs[i] + hypothesis[range(len(log_trans_mat)), backtrack[i]]

        path = [torch.argmax(omega + self.final_probs.log())]
        for i in reversed(range(1, len(llhs))):
            path.insert(0, backtrack[i, path[0]])
        return torch.LongTensor(path)

    def float(self):
            return CompiledGraph(self.init_probs.float(),
                                 self.final_probs.float(),
                                 self.trans_probs.float(),
                                 self.pdf_id_mapping)

    def double(self):
        return CompiledGraph(self.init_probs.double(),
                                 self.final_probs.double(),
                                 self.trans_probs.double(),
                                 self.pdf_id_mapping)

    def to(self, device):
        return CompiledGraph(self.init_probs.to(device),
                                 self.final_probs.to(device),
                                 self.trans_probs.to(device),
                                 self.pdf_id_mapping)


__all__ = ['Graph']
