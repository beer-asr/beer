'Acoustic graph for the HMM.'

from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Set, Dict, TypeVar, Generic
import torch
from .utils import logsumexp

StateType = TypeVar('StateType')
ArcType = TypeVar('ArcType')


@dataclass
class State(Generic[StateType]):
    'State (i.e. node) of a graph.'
    id: int
    pdf_id: int


@dataclass(unsafe_hash=True)
class Arc(Generic[ArcType]):
    'Arc between 2 states (i.e. node) of a graph with a weight.'
    start: int
    end: int
    weight: float = field(compare=False)


def _state_name(symbols, state_id):
    try:
        name = str(symbols[state_id])
    except KeyError:
        name = str(state_id)
    return name


def _show_graph(graph):
    # We import the module here as it is only needed by the Jupyter
    # notebook.
    import graphviz
    dot = graphviz.Digraph()
    dot.graph_attr['rankdir'] = 'LR'
    for state_id in graph.states():
        attrs = {'shape': 'circle'}
        if state_id == graph.start_state:
            attrs.update(penwidth='2.0')
        if state_id == graph.end_state:
            attrs.update(shape='doublecircle')
        dot.node(_state_name(graph.symbols, state_id), **attrs)
    for arc in graph.arcs():
        dot.edge(_state_name(graph.symbols, arc.start),
                 _state_name(graph.symbols, arc.end),
                 label=str(round(arc.weight, 3)))
    return graphviz.Source(dot.source)._repr_svg_()


@dataclass
class Graph:

    _state_count: int = field(default=0, init=False, repr=False)
    _states: Dict[int, StateType] = field(default_factory=OrderedDict, init=False,
                                          repr=False)
    _arcs: Set[ArcType] = field(default_factory=set, init=False, repr=False)
    symbols: Dict[int, str] = field(default_factory=dict, init=False,
                                    repr=False)
    start_state: int = field(default=None, init=False, repr=False)
    end_state: int = field(default=None, init=False, repr=False)

    def _repr_svg_(self):
        return _show_graph(self)

    def states(self):
        'Iterator over the states.'
        return self._states.keys()

    def state_from_id(self, state_id):
        return self._states[state_id]

    def arcs(self, state_id=None, incoming=False):
        '''Iterator over the arcs.

        Args:
            state_id (int): If provided, consider only the incoming/
                outgoing arcs from the given states.
            incomping (boolean): If True/False, only iterates over the
                incoming/outgoing arcs. This field is used only if
                state_id is provided.

        Yields:
            ``Arc``.
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
        self._states[new_state.id] = new_state
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

    def find_next_pdf_ids(self, start_state, init_weight=1.0):
        to_explore = [(arc, init_weight) for arc in self.arcs(start_state)]
        visited = set([start_state])
        while to_explore:
            arc, weight = to_explore.pop()
            pdf_id = self._states[arc.end].pdf_id
            if pdf_id is not None:
                yield arc.end, weight * arc.weight
            else:
                if arc.end not in visited:
                    to_explore += [(arc, arc.weight * weight)
                                    for arc in self.arcs(arc.end)]
                    visited.add(arc.end)

    def find_previous_pdf_ids(self, start_state, init_weight=1.0):
        to_explore = [(arc, init_weight)
                      for arc in self.arcs(start_state, incoming=True)]
        visited = set([start_state])
        while to_explore:
            arc, weight = to_explore.pop()
            pdf_id = self._states[arc.start].pdf_id
            if pdf_id is not None:
                yield arc.start, weight * arc.weight
            else:
                if arc.start not in visited:
                    to_explore += [(arc, arc.weight * weight)
                                    for arc in self.arcs(arc.start, incoming=True)]
                    visited.add(arc.start)

    def compile(self):
        'Compile the graph.'

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
        for state_id, weight in self.find_next_pdf_ids(self.start_state, 1.0):
            init_probs[state2pdf_id[state_id]] += weight
        init_probs /= init_probs.sum()

        # Init probs.
        for state_id, weight in self.find_previous_pdf_ids(self.end_state, 1.0):
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
                for state_id, weight in self.find_next_pdf_ids(arc.end, weight):
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

        return CompiledGraph(init_probs.log(), final_probs.log(),
                             trans_probs.log(), pdf_id_mapping)


class CompiledGraph:
    '''Inference graph for a HMM model.'''

    def __init__(self, init_log_probs, final_log_probs, trans_log_probs,
                 pdf_id_mapping=None):
        '''
        Args:
            init_log_probs (``torch.Tensor``): Initial log probabilities.
            final_log_probs (``torch.Tensor``): Final log probabilities.
            trans_log_probs (``torch.Tensor``): Transition log probabilities.
            pdf_id_mapping (list): Mapping of the pdf ids (optional)
        '''
        self.init_log_probs = init_log_probs
        self.final_log_probs = final_log_probs
        self.trans_log_probs = trans_log_probs
        self.pdf_id_mapping = pdf_id_mapping

    @property
    def n_states(self):
        'Total number of states in the graph.'
        return len(self.trans_log_probs)

    def _baum_welch_forward(self, llhs):
        log_trans_mat = self.trans_log_probs
        log_alphas = torch.zeros_like(llhs) - float('inf')
        log_alphas[0] = llhs[0] + self.init_log_probs
        for i in range(1, llhs.shape[0]):
            log_alphas[i] = llhs[i]
            log_alphas[i] += torch.logsumexp(log_alphas[i-1] + log_trans_mat.t(),
                                             dim=1).view(-1)
        return log_alphas

    def _baum_welch_backward(self, llhs):
        log_trans_mat = self.trans_log_probs
        log_betas = torch.zeros_like(llhs) - float('inf')
        log_betas[-1] = self.final_log_probs
        for i in reversed(range(llhs.shape[0]-1)):
            log_betas[i] = torch.logsumexp(log_trans_mat + llhs[i+1] + \
                           log_betas[i+1], dim=1).view(-1)
        return log_betas

    def posteriors(self, llhs, trans_posteriors=False):
        '''Compute the posterior of the state given the
        (log-)likelihood of the data.

        Args:
            llhs (``torch.Tensor[N, K]``): Log-likelihood per frame and
                state.
            trans_posteriors (boolean): If true, also compute the
                transition posterior.

        Returns:
            ``torch.FloatTensor[N, K]``: state posteriors.
            ``torch.FloatTensor[N-1, K, K]``: transition posteriors

        '''
        log_alphas = self._baum_welch_forward(llhs)
        log_betas = self._baum_welch_backward(llhs)
        lognorm = torch.logsumexp((log_alphas + log_betas)[0], dim=0)
        state_posts = (log_alphas + log_betas - lognorm).exp()
        if trans_posteriors:
            log_A = self.trans_log_probs
            log_xi = log_alphas[:-1, :, None] + log_A[None] + \
                     (llhs + log_betas)[1:, None, :]
            log_xi = log_xi.view(-1, len(log_A) * len(log_A))
            lnorm = torch.logsumexp(log_xi[0], dim=0)
            trans_posts = (log_xi - lnorm).exp()
            trans_posts = torch.where(trans_posts != trans_posts,
                                     torch.zeros_like(trans_posts),
                                     trans_posts)
            trans_posts = trans_posts.view(-1, len(log_A), len(log_A))
            retval = state_posts, trans_posts
        else:
            retval = state_posts
        return retval


    def best_path(self, llhs):
        init_log_prob = self.init_log_probs
        backtrack = torch.zeros_like(llhs, dtype=torch.long,
                                     device=llhs.device)
        omega = llhs[0] + init_log_prob
        log_trans_mat = self.trans_log_probs

        for i in range(1, llhs.shape[0]):
            hypothesis = omega + log_trans_mat.t()
            backtrack[i] = torch.argmax(hypothesis, dim=1)
            omega = llhs[i] + hypothesis[range(len(log_trans_mat)), backtrack[i]]

        path = [torch.argmax(omega + self.final_log_probs)]
        for i in reversed(range(1, len(llhs))):
            path.insert(0, backtrack[i, path[0]])
        return torch.LongTensor(path)

    def float(self):
            return CompiledGraph(self.init_log_probs.float(),
                                 self.final_log_probs.float(),
                                 self.trans_log_probs.float(),
                                 self.pdf_id_mapping)

    def double(self):
        return CompiledGraph(self.init_log_probs.double(),
                                 self.final_log_probs.double(),
                                 self.trans_log_probs.double(),
                                 self.pdf_id_mapping)

    def to(self, device):
        return CompiledGraph(self.init_log_probs.to(device),
                                 self.final_log_probs.to(device),
                                 self.trans_log_probs.to(device),
                                 self.pdf_id_mapping)


__all__ = ['Graph']

