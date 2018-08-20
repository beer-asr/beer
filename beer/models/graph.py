'Acoustic graph for the HMM.'

import torch
from ..utils import logsumexp

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
    __repr_str = 'State(unit_id={}, state_id={}, pdf_id={})'

    def __init__(self, unit_id, state_id, pdf_id):
        self.unit_id = unit_id
        self.state_id = state_id
        self.pdf_id = pdf_id

    @property
    def name(self):
        return str(self.unit_id) + '_' + str(self.state_id)

    def __hash__(self):
        return hash('{}:{}'.format(self.unit_id, self.state_id))

    def __repr__(self):
        return self.__repr_str.format(self.unit_id, self.state_id, self.pdf_id)

    def __eq__(self, other):
        if isinstance(other, State):
            return hash(self) == hash(other)
        return NotImplemented


class Graph:
    '''Graph.

    Attributes:
        states (dictionary): All the states of the graph.
        arcs (dictionary): All the arcs of the graph.
    '''

    def __init__(self):
        start = State(unit_id='<s>', state_id=0, pdf_id=None)
        end = State(unit_id='</s>', state_id=0, pdf_id=None)
        self._states = {
            start.name: start, end.name: end
        }
        self._arcs = set()

    @property
    def start_state(self):
        return self._states['<s>_0']

    @property
    def end_state(self):
        return self._states['</s>_0']

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
        for state in self._states.values():
            attrs = {'shape': 'circle'}
            if state.unit_id == '<s>' or state.unit_id == '</s>':
                attrs.update(shape='point')
            dot.node(state.name, **attrs)
        for arc in self.arcs():
            dot.edge(str(arc.start), str(arc.end), label=str(round(arc.weight, 3)))
        return graphviz.Source(dot.source)._repr_svg_()

    def arcs(self, state=None, outgoing=True):
        '''Iterates over the arcs. If state is provided enumerate the
        outgoing args from "state"
        '''
        for arc in self._arcs:
            if outgoing:
                if state is None or arc.start == state.name:
                    yield arc
            else:
                if state is None or arc.end == state.name:
                    yield arc

    def add_state(self, unit_id, state_id, pdf_id):
        new_state = State(unit_id, state_id, pdf_id)
        self._states[new_state.name] = new_state
        return new_state

    def add_arc(self, start, end, weight):
        new_arc = Arc(start.name, end.name, weight)
        self._arcs.add(new_arc)
        return new_arc

    def join(self, graphs):
        'Concatenate a set of graphs into one big graph.'
        for graph in graphs:
            self.add_graph(graph)

    def add_graph(self, graph):
        self._states.update(graph._states)
        self._arcs.update(graph._arcs)

    def normalize(self):
        for state in self._states.values():
            sum_out_weights = 0.
            for arc in self.arcs(state):
                sum_out_weights += arc.weight
            for arc in self.arcs(state):
                arc.weight /= sum_out_weights

    def to_matrix(self):
        nstates = len(self.states)
        matrix = torch.zeros((nstates, nstates))
        for arc in self.arcs.values():
            matrix[arc.start, arc.end] = arc.weight
        return matrix

    def _find_next_pdf_ids(self, start_state, init_weight):
        for arc in self.arcs(start_state):
            pdf_id = self._states[arc.end].pdf_id
            if pdf_id is not None:
                yield pdf_id, init_weight * arc.weight
            else:
                yield from self._find_next_pdf_ids(self._states[arc.end],
                                                   init_weight * arc.weight)

    def compile(self):
        '''Compile the graph.'''
        # Total number of states
        tot_n_states = len(self._states) - 2
        init_probs = torch.zeros(tot_n_states)
        final_probs = torch.zeros(tot_n_states)
        trans_probs = torch.zeros(tot_n_states, tot_n_states)

        # Init probs.
        for arc in self.arcs(state=self.start_state):
            pdf_id = self._states[arc.end].pdf_id
            init_probs[pdf_id] = arc.weight

        # Init probs.
        for arc in self.arcs(state=self.end_state, outgoing=False):
            pdf_id = self._states[arc.start].pdf_id
            final_probs[pdf_id] = arc.weight
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
            if pdf_id2 is None:
                for pdf_id2, weight in self._find_next_pdf_ids(self._states[arc.end],
                                                               weight):
                    trans_probs[pdf_id1, pdf_id2] = weight
            else:
                trans_probs[pdf_id1, pdf_id2] = weight

        return CompiledGraph(init_probs, final_probs, trans_probs)

class CompiledGraph:
    '''Inference graph for a HMM model.'''

    def __init__(self, init_probs, final_probs, trans_probs):
        '''
        Args:
            init_probs (``torch.Tensor``): Initial probabilities.
            final_probs (``torch.Tensor``): Final probabilities.
            trans_probs (``torch.Tensor``): Transition probabilities.
        '''
        self.init_probs = init_probs
        self.final_probs = final_probs
        self.trans_probs = trans_probs

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
        lhs = (llhs - max_val).exp() + 1e-6

        # Scaled forward-backward algorithm.
        alphas, consts = self._baum_welch_forward(lhs)
        betas = self._baum_welch_backward(lhs, consts)
        posts = alphas * betas
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



__all__ = ['Graph']
