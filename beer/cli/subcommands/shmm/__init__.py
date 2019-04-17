'Subspace Hidden Markov Model (SHMM)'

from . import mksphoneloop
from . import train

cmds = [mksphoneloop, train]

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

