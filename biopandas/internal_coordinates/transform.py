
import numpy as np 
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist


def calculate_dihedral(p0, p1, p2, p3): 
    """Returns a torsion angle between four points. 

    Parameters
    ----------
    p0 : np.array
        coordinate one 
    p1 : np.array 
        coordinate two 
    p2 : np.array
        coordinate three
    p3 : np.array 
        coordinate four 

    Returns
    ---------
    torsion_angle : float
        angle between the four points (the two planes
        defined by the first set and second set of 
        points)
    """

    # calculate vectors between points 
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    
    # normalize b1
    b1 /= np.linalg.norm(b1)
    
    # calculate vector rejections
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1
    
    # calculate angle between v and w
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    
    return np.arctan2(y, x)

def calculate_phiomgpsis(df): 
    """Returns the backbone torsion angles (ordered as psi_i, omega_i, phi_i)

    Parameters
    ----------
    df : biopandas.pdb.PandasPdb
        PandasPdb object containing structure information 

    Returns
    ---------
    phiomgpsis : np.array 
        a np.array containing the torsion angles for the 
        structure (ordered as psi_i, omega_i, phi_i)
    """

    # initialize storage
    phiomgpsis = list()

    # get only atoms that are relevant for the calculation 
    atoms = ["N", "CA", "C"]
    use_atom = np.zeros(df.shape[0])
    for atom in atoms: 
        use_atom = np.logical_or(use_atom, (df.atom_name == atom).to_numpy())
    df = df[use_atom].reset_index().drop(columns="index")

    # get indices of the "C" backbone atoms 
    idxs = df.loc[df.atom_name == "C"].index

    # iterate through the backbone atoms 
    for i,k in enumerate(idxs): 

        if i >= idxs.shape[0]-1: 
            continue

        # grab relevant coordinates for calculation 
        p0, p1, p2, p3, p4, p5 = df.iloc[k-2:k+4][["x_coord", "y_coord", "z_coord"]].to_numpy()

        # calculate psi
        psi = calculate_dihedral(p0, p1, p2, p3)

        # calculate omega
        omega = calculate_dihedral(p1, p2, p3, p4)

        # calculate phi 
        phi = calculate_dihedral(p2, p3, p4, p5)

        phiomgpsis.extend([psi, omega, phi])

    return np.array(phiomgpsis)

def calculate_bond_distances(df): 
    """Returns the backbone bond distances.

    Parameters
    ----------
    df : biopandas.pdb.PandasPdb
        PandasPdb object containing structure information 

    Returns
    ---------
    bond_distances : np.array 
        a np.array containing the bond distances for the structure 
    """

    # get the desired coordinates for the current structure 
    atoms = ["N", "CA", "C"]
    use_atom = np.zeros(df.shape[0])
    for atom in atoms: 
        use_atom = np.logical_or(use_atom, (df.atom_name == atom).to_numpy())
    df = df[use_atom].reset_index().drop(columns="index")
    coords = df[["x_coord", "y_coord", "z_coord"]].to_numpy()

    # calculate the consecutive bond distances 
    bond_distances = np.diag(cdist(coords, coords), k=1)[2:]

    return np.array(bond_distances).astype("float32")

def calculate_bond_angles(df): 
    """Returns the backbone bond angles.

    Parameters
    ----------
    df : biopandas.pdb.PandasPdb
        PandasPdb object containing structure information 

    Returns
    ---------
    bond_angles : np.array 
        a np.array containing the bond angles for the structure 
    """

    # get the desired coordinates for the current structure 
    atoms = ["N", "CA", "C"]
    use_atom = np.zeros(df.shape[0])
    for atom in atoms: 
        use_atom = np.logical_or(use_atom, (df.atom_name == atom).to_numpy())
    df = df[use_atom].reset_index().drop(columns="index")
    coords = df[["x_coord", "y_coord", "z_coord"]].to_numpy()

    # calculate and normalize the difference vectors between coordinates
    firsts = normalize(coords[1:-2] - coords[2:-1], axis=1)
    seconds = normalize(coords[3:] - coords[2:-1], axis=1)

    # initialize storage array for calculation 
    bond_angles = list()

    # calculate the angles between the difference vectors 
    for i in np.arange(firsts.shape[0]): 

        bond_angles.append(np.arccos(np.clip(np.dot(firsts[i,:], seconds[i,:]), -1.0, 1.0)))

    return np.array(bond_angles).astype("float32")

def reconstruction(initial_coords, bond_distances, bond_angles, torsion_angles): 
    """Implemenatation of protein backbone reconstruction with internal coordinates 
    based on the Natural Extension Reference Frame (NERF).
    Parsons et al., 2005, Journal of Computational Chemistry 

    Parameters
    ----------
    initial_coords : np.array 
        3 x 3 np.array containing the coordinates of the first three
        atoms of the protein 
    bond_distances : np.array 
        a np.array containing the bond distances for the structure 
    bond_angles : np.array 
        a np.array containing the bond angles for the structure 
    torsion_angles : np.array 
        a np.array containing the torsion angles for the 
        structure (ordered as psi_i, omega_i, phi_i)

    Returns
    ---------
    reconstructed_coordinates : np.array 
        a np.array containing the reconstructed coordinates of the protein
    """

    # initialize a list for collecting coordinates and dealing with coordinates 
    coord_list = list(initial_coords.flatten())
    current_coords = initial_coords # will be updated after each pass 

    # make sure that the reconstruction parameters are all of the same length 
    assert(bond_angles.shape == bond_distances.shape)
    assert(bond_angles.shape == torsion_angles.shape)
    assert(bond_distances.shape == torsion_angles.shape)

    # use the supplement of the bond angles 
    bond_angles = np.pi - bond_angles

    # iterate throught the supplied data to reconstruct the backbone 
    for i in np.arange(bond_angles.shape[0]): 

        # calculate the D_2 vector 
        D2 = bond_distances[i] * np.array([np.cos(bond_angles[i]), np.cos(torsion_angles[i]) * np.sin(bond_angles[i]), np.sin(torsion_angles[i]) * np.sin(bond_angles[i])])

        # calculate useful intermediate values 
        bc = current_coords[2,:] - current_coords[1,:]
        bc_hat = bc / np.linalg.norm(bc)
        ab = current_coords[1,:] - current_coords[0,:]
        n = np.cross(ab, bc_hat)
        n_hat = n / np.linalg.norm(n)
        tmp = np.cross(n_hat, bc_hat) / np.linalg.norm(np.cross(n_hat, bc_hat))

        # calculate the transformation matrix M
        M = np.hstack([bc_hat[:,None], tmp[:,None], n_hat[:,None]]).T

        # calculate the coordinate D and extend coord_list
        D = D2 @ M + current_coords[2,:]
        coord_list.extend(D)

        # update the current_coords
        current_coords = np.vstack([current_coords[1:,:], D])

    return np.array(coord_list).astype("float64").reshape(-1, 3)

class BackboneTransform(object): 
    """
    Object for working transforming backbone xyz coordinates to and from internal coordinates. 

    Attributes
    ----------
    df : pd.DataFrame
        a pd.DataFrame containing 'ATOM' records of the desired 
        structure 

    initial_coords : np.array 
        3 x 3 np.array containing the coordinates of the first three
        atoms of the protein 

    """

    def __init__(self, PandasPdb_obj): 
        self.df = PandasPdb_obj.df["ATOM"]
        self.initial_coords = self.df.iloc[:3][["x_coord", "y_coord", "z_coord"]].to_numpy()

    def xyz2ic(self): 
        """Conversion of xyz coordinates to internal coordinates. 

        Parameters
        ----------
        None

        Returns
        ---------
        bond_distances : np.array 
            a np.array containing the bond distances for the structure 
        bond_angles : np.array 
            a np.array containing the bond angles for the structure 
        torsion_angles : np.array 
            a np.array containing the torsion angles for the 
            structure (ordered as psi_i, omega_i, phi_i)
        """

        # calculate the internal coordinates of the model 
        bond_distances = calculate_bond_distances(self.df)
        bond_angles = calculate_bond_angles(self.df)
        torsion_angles = calculate_phiomgpsis(self.df)

        return bond_distances, bond_angles, torsion_angles

    def ic2xyz(self, bond_distances, bond_angles, torsion_angles): 
        """Conversion of internal coordinates to xyz coordinates. 

        Parameters
        ----------
        bond_distances : np.array 
            a np.array containing the bond distances for the structure 
        bond_angles : np.array 
            a np.array containing the bond angles for the structure 
        torsion_angles : np.array 
            a np.array containing the torsion angles for the 
            structure (ordered as psi_i, omega_i, phi_i)

        Returns
        ---------
        reconstructed_coordinates : np.array 
            a np.array containing the reconstructed coordinates of the model
        """

        # reconstruct the model using the supplied internal coordinates
        coordinates = reconstruction(initial_coords=self.initial_coords, bond_distances=bond_distances, bond_angles=bond_angles, torsion_angles=torsion_angles)

        return coordinates