
'Acoustic graph for the HMM.'

import torch
import numpy as np


class Arc:
    __repr_str = 'Arc(arc_id={}, start={}, end={}, weight={})'

    def __init__(self, arc_id, start, end, weight):
        '''
        Args:
            arc_id (int): Unique arc number.
            start (int): Starting state.
            end (int): Final state,
            weight (int): Weight of the arc.
        '''
        self.arc_id = arc_id
        self.start = start
        self.end = end
        self.weight = weight

    def __repr__(self):
        return self.__repr_str.format(self.arc_id, self.start, self.end,
                                      self.weight)


class State:
    __repr_str = 'State(state_id={}, init_weight={}, final_weight={})'

    def __init__(self, state_id):
        self.init_weight = -1.
        self.final_weight = -1
        self.state_id = state_id
        self.arcs = {}

    def __repr__(self):
        retval = self.__repr_str.format(self.state_id, self.init_weight,
                                        self.final_weight)
        for i, arc_id in enumerate(self.arcs):
            retval += '\n'
            retval += '  (' + str(i + 1) + ') ' + repr(self.arcs[arc_id])
        return retval

class Graph:
    def __init__(self):
        self.states = {}
        self.arcs = {}

    def __repr__(self):
        retval = ''
        for i, state in enumerate(self.states.values()):
            retval += repr(state)
            if i <= len(self.states) - 1:
                retval += '\n'
        return retval

    def add_state(self):
        state_id = len(self.states)
        new_state = State(state_id)
        self.states[state_id] = new_state
        return state_id

    def add_arc(self, start, end, weight):
        arc_id = len(self.arcs)
        new_arc = Arc(arc_id, start, end, weight)
        self.arcs[arc_id] = new_arc
        self.states[start].arcs[arc_id] = new_arc
        return arc_id

    def initial_states(self):
        for state_id, state in self.states.items():
            if state.init_weight > 0:
                yield state_id

    def final_states(self):
        for state_id, state in self.states.items():
            if state.final_weight > 0:
                yield state_id

    def to_matrix(self):
        nstates = len(self.states)
        matrix = torch.zeros((nstates, nstates))
        for arc in self.arcs.values():
            matrix[arc.start, arc.end] = arc.weight
        return matrix


class AcousticGraph(Graph):

    def __init__(self):
        super().__init__()
        self.units = {}

    def add_unit(self, n_states, edges_conf):
        state_ids = [self.add_state() for i in range(n_states)]
        self.states[state_ids[0]].init_weight = 1.0
        for edge_conf in edges_conf:
            if edge_conf['end_id'] == '<exit>':
                self.states[state_ids[-1]].final_weight = edge_conf['trans_prob']
            else:
                start = state_ids[edge_conf['start_id']]
                end = state_ids[edge_conf['end_id']]
                weight = edge_conf['trans_prob']
                self.add_arc(start, end, weight)
        unit_id = len(self.units)
        self.units[unit_id] = state_ids

    def to_matrix(self):
        matrix = super().to_matrix()
        prob_unit = 1. / len(self.units)
        for state_id in self.final_states():
            final_weight = self.states[state_id].final_weight
            for unit_states in self.units.values():
                matrix[state_id, unit_states[0]] = final_weight * prob_unit
        return matrix


__all__ = ['AcousticGraph']
