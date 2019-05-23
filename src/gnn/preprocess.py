#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import click
from ruamel.yaml import YAML
from pathlib import Path
from gnn.data_utils import *
from sklearn.model_selection import train_test_split


@click.command()
@click.option('--data-cnf', help='dataset config')
def main(data_cnf):
    yaml = YAML(typ='safe')
    data_cnf = yaml.load(Path(data_cnf))
    dataset, radius, ngram = data_cnf['name'], data_cnf['model']['radius'], data_cnf['model']['ngram']

    with open(data_cnf['source'][0], 'r') as f:
        data_list_train = f.read().strip().split('\n')
    with open(data_cnf['source'][1], 'r') as f:
        data_list_valid = f.read().strip().split('\n')

    """Exclude data contains '.' in the SMILES format."""
    data_list_train = [d for d in data_list_train if '.' not in d.strip().split()[0]]
    data_list_valid = [d for d in data_list_valid if '.' not in d.strip().split()[0]]

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    word_dict = defaultdict(lambda: len(word_dict))

    # train data
    smiles, compounds, adjacencies, proteins, interactions = '', [], [], [], []

    failed = 0

    for no, data in enumerate(data_list_train):
        print('/'.join(map(str, [no + 1, len(data_list_train)])))

        try:
            smile, sequence, interaction = data.strip().split()
            smiles += smile + '\n'

            mol = Chem.AddHs(Chem.MolFromSmiles(smile))
            atoms = create_atoms(mol, atom_dict)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)

            fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius, fingerprint_dict, edge_dict)
            compounds.append(fingerprints)

            adjacency = create_adjacency(mol)
            adjacencies.append(adjacency)

            words = split_sequence(sequence, ngram, word_dict)
            proteins.append(words)

            interactions.append(np.array([float(interaction)]))
        except Exception:
            failed += 1

    print('failed:', failed)
    train_x = np.asarray(list(zip(compounds, adjacencies, proteins)))
    train_y = np.asarray(interactions)

    # valid data
    smiles, compounds, adjacencies, proteins, interactions = '', [], [], [], []

    for no, data in enumerate(data_list_valid):
        print('/'.join(map(str, [no + 1, len(data_list_valid)])))

        try:
            smile, sequence, interaction = data.strip().split()
            smiles += smile + '\n'

            mol = Chem.AddHs(Chem.MolFromSmiles(smile))
            atoms = create_atoms(mol, atom_dict)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)

            fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius, fingerprint_dict, edge_dict)
            compounds.append(fingerprints)

            adjacency = create_adjacency(mol)
            adjacencies.append(adjacency)

            words = split_sequence(sequence, ngram, word_dict)
            proteins.append(words)

            interactions.append(np.array([float(interaction)]))
        except Exception:
            failed += 1

    print('failed:', failed)
    valid_x = np.asarray(list(zip(compounds, adjacencies, proteins)))
    valid_y = np.asarray(interactions)

    np.save(data_cnf['train']['input'], train_x)
    np.save(data_cnf['train']['label'], train_y)
    np.save(data_cnf['valid']['input'], valid_x)
    np.save(data_cnf['valid']['label'], valid_y)

    dump_dictionary(fingerprint_dict, data_cnf['fingerprint_dict'])
    dump_dictionary(word_dict, data_cnf['word_dict'])

    print('The preprocess of ' + dataset + ' dataset has finished!')


if __name__ == "__main__":
    main()
