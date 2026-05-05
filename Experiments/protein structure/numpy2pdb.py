import numpy as np
from Bio.PDB import PDBParser, PDBIO


def numpy2pdb(frame, backbone_structure_file, output_file):

    frame = frame.reshape(-1, 3)
    structure = PDBParser(QUIET=True).get_structure("x", backbone_structure_file)
    model = structure[0]

    atoms = list(model.get_atoms())[0:len(frame)]  


    for atom, xyz in zip(atoms, frame):
        atom.set_coord(xyz)

    io = PDBIO()
    io.set_structure(model)   
    io.save(output_file)
        
    