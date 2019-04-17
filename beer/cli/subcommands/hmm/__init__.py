'Hidden Markov Model (HMM)'

from . import accumulate
from . import decode
from . import mkaligraph
from . import mkdecodegraph
from . import mkphoneloop
from . import mkphoneloopgraph
from . import mkphones
from . import optimizer
from . import posteriors
from . import phonelist
from . import train
from . import update


cmds = [accumulate, decode, mkaligraph, mkdecodegraph, mkphoneloop,
        mkphoneloopgraph, mkphones, optimizer, posteriors,
        phonelist, train, update]

def setup(parser):
    subparsers = parser.add_subparsers(title='possible commands', metavar='<cmd>')
    subparsers.required = True
    for cmd in cmds:
        cmd_name = cmd.__name__.split('.')[-1]
        subparser = subparsers.add_parser(cmd_name, help=cmd.__doc__)
        cmd.setup(subparser)
        subparser.set_defaults(func=cmd.main)


def main(args, logger):
    pass

