from pymol.cgo import *
from pymol import cmd
VORONOI_CONTACTS = [COLOR, 1.000, 1.000, 1.000,
ALPHA, 1.000,
]
cmd.load_cgo(VORONOI_CONTACTS, 'VORONOI_CONTACTS')
cmd.set('two_sided_lighting', 'on')
