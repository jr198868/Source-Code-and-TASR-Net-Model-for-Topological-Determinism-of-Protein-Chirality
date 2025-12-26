import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
from gudhi.representations import Landscape, BettiCurve
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import seaborn as sns
from scipy.spatial.distance import cdist
import gudhi
import logging
import joblib
import glob
from functools import partial
#F3DR: Feature‑to‑3D Regression

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================
# Dynamic path configuration
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================
# Configuration parameters
# =====================
DEFAULT_CONFIG = {
    "tda_csv_path": os.path.join(PROJECT_ROOT, "Research_results/step3.2_all_barcode_points/step3.2_all_barcode_points.csv"),
    "coords_dir": os.path.join(PROJECT_ROOT, "Research_results/step2.0_Pretreatment_results/step2.0_aligned_coords"),
    "pca_variance": 0.95,
    "max_ca_atoms": 200,
    "test_size": 0.2,
    "random_state": 42,
    "output_dir": os.path.join(PROJECT_ROOT, "Research_results/Step4.1_and_4.2_topological_signature_modeling"),
    "vectorization_method": "multiscale",
    "batch_size": 32,
    "epochs": 100,
    "latent_dim": 256,
    "chunk_size": 2000  # Number of proteins to process per batch
}

# =====================
# Data loading and processing
# =====================

def load_and_process_data(config, start_idx=0, end_idx=None):
    """Load and process data, including creating mirrored structures as control group"""
    logger.info("Starting to load data...")
    
    # 1. LoadingTDA
    try:
        barcode_df = pd.read_csv(config['tda_csv_path'])
        logger.info(f"Successfully loaded TDA data, total{barcode_df.shape[0]}rows")
    except Exception as e:
        logger.error(f"Failed to load TDA data: {e}")
        return []
    
    # 2. Processing TDA data by groups
    grouped = barcode_df.groupby(['ID', 'dim'])
    tda_data = {}
    
    # Getting unique ID list
    all_ids = sorted(barcode_df['ID'].unique())
    if end_idx is None or end_idx > len(all_ids):
        end_idx = len(all_ids)
    selected_ids = all_ids[start_idx:end_idx]
    logger.info(f"Selected to process{len(selected_ids)}proteins (index{start_idx}-{end_idx-1})")
    
    # Processing TDA data
    for (id, dim), group in tqdm(grouped, desc="Processing TDA data"):
        if id not in selected_ids:
            continue
        if id not in tda_data:
            tda_data[id] = {0: [], 1: [], 2: []}
        points = group[['birth', 'death']].values
        tda_data[id][dim] = points
    
    # 3. Loading coordinate data
    coord_data = {}
    coord_files = [f for f in os.listdir(config['coords_dir']) if f.endswith('_aligned_coords.txt')]
    logger.info(f"in{config['coords_dir']}directory found{len(coord_files)}coordinate files")
    
    # Creating ID to filename mapping
    file_id_map = {}
    for file in coord_files:
        protein_id = file.replace('_aligned_coords.txt', '')
        file_id_map[protein_id] = file

    # Loading coordinates
    for protein_id in tqdm(selected_ids, desc="Loading coordinates"):
        if protein_id in file_id_map:
            file_path = os.path.join(config['coords_dir'], file_id_map[protein_id])
            try:
                # 
                if os.path.getsize(file_path) == 0:
                    logger.warning(f" {file_path}is empty, using zero padding")
                    coords = np.zeros((config['max_ca_atoms'], 3))
                else:
                    coords = np.loadtxt(file_path)
                
                # Ensure coordinates are 2D array
                if coords.size == 0:  # Loading
                    logger.warning(f" {file_path}is empty after loading, using zero padding")
                    coords = np.zeros((config['max_ca_atoms'], 3))
                elif coords.ndim == 1:
                    # , (1, 3)
                    coords = coords.reshape(1, -1)
                    logger.warning(f" {protein_id}coordinate is 1D array, converted to 2D")
                
                coord_data[protein_id] = coords
            except Exception as e:
                logger.error(f"Loading{file_path}failed: {e}")
                # 
                coord_data[protein_id] = np.zeros((config['max_ca_atoms'], 3))
        else:
            logger.warning(f" {protein_id}coordinate file does not exist, using zero padding")
            coord_data[protein_id] = np.zeros((config['max_ca_atoms'], 3))
    
    # 4. Creating mirrored structure control group
    mirrored_data = []
    for pid, coords in coord_data.items():
        # Ensure coordinates are 2D array
        if coords.ndim == 1:
            # , (1, 3)
            coords = coords.reshape(1, -1)
            logger.warning(f" {pid}coordinate is 1D array, converted to 2D")
        
        # 
        mirrored_coords = coords.copy()
        if mirrored_coords.shape[1] >= 1:  # 
            mirrored_coords[:, 0] = -mirrored_coords[:, 0]  # X
        else:
            logger.error(f" {pid}coordinate dimension is insufficient, cannot create mirrored structure")
            continue
            
        mirrored_data.append({
            'id': pid + "_mirrored",
            'coords': mirrored_coords
        })
    
    # 5. Integrating data
    final_data = []
    for pid in coord_data.keys():
        if pid in tda_data:
            final_data.append({
                'id': pid,
                'tda_dim0': tda_data[pid][0] if pid in tda_data else np.empty((0, 2)),
                'tda_dim1': tda_data[pid][1] if pid in tda_data else np.empty((0, 2)),
                'tda_dim2': tda_data[pid][2] if pid in tda_data else np.empty((0, 2)),
                'coords': coord_data[pid]
            })
    
    # Adding mirrored structures
    for mirror in mirrored_data:
        # TDA
        final_data.append({
            'id': mirror['id'],
            'tda_dim0': np.empty((0, 2)),
            'tda_dim1': np.empty((0, 2)),
            'tda_dim2': np.empty((0, 2)),
            'coords': mirror['coords']
        })
    
    logger.info(f"Final dataset contains{len(final_data)}proteins (including{len(mirrored_data)}mirrored structures)")
    
    return final_data

# =====================
# Feature engineering
# =====================

def vectorize_tda_features(data, config):
    """Multiscale topological feature extraction"""
    logger.info("Starting TDA feature vectorization...")
    
    # Feature generator configuration
    feature_generators = [
        Landscape(num_landscapes=5, resolution=50),
        BettiCurve(resolution=100)
    ]
    
    # Computing feature length for each dimension
    dim_feature_length = 0
    for generator in feature_generators:
        if isinstance(generator, Landscape):
            dim_feature_length += generator.num_landscapes * generator.resolution
        elif isinstance(generator, BettiCurve):
            dim_feature_length += generator.resolution
    dim_feature_length += 4  # Statistical features
    
    # Total feature length per protein (3 dimensions)
    total_feature_length = dim_feature_length * 3
    logger.info(f"Feature length per dimension: {dim_feature_length}, Total feature length: {total_feature_length}")
    
    for protein in tqdm(data, desc="Feature engineering"):
        all_features = []
        
        for dim in [0, 1, 2]:
            # Ensure tda_dim exists
            points_key = f'tda_dim{dim}'
            points = protein.get(points_key, np.empty((0, 2)))
            
            # ,
            if isinstance(points, list):
                points = np.array(points)
            
            dim_features = []
            
            # Applying feature generators
            for generator in feature_generators:
                if len(points) > 0 and points.shape[1] == 2:
                    try:
                        feat = generator.fit_transform([points])[0]
                    except Exception as e:
                        logger.warning(f"{protein['id']} dim{dim}Feature generation failed: {e}")
                        # Creating zero vector of appropriate length
                        if isinstance(generator, Landscape):
                            feat = np.zeros(generator.num_landscapes * generator.resolution)
                        else:  # BettiCurve
                            feat = np.zeros(generator.resolution)
                else:
                    # Creating zero vector of appropriate length
                    if isinstance(generator, Landscape):
                        feat = np.zeros(generator.num_landscapes * generator.resolution)
                    else:  # BettiCurve
                        feat = np.zeros(generator.resolution)
                
                # 
                if feat.ndim == 0:
                    feat = np.array([feat])
                elif feat.ndim > 1:
                    feat = feat.flatten()
                
                dim_features.append(feat)
            
            # Adding statistical features
            if len(points) > 0 and points.shape[1] == 2:
                lifetimes = points[:, 1] - points[:, 0]
                stat_features = np.array([
                    np.mean(lifetimes),
                    np.max(lifetimes),
                    np.min(lifetimes),
                    np.percentile(lifetimes, 90)
                ])
            else:
                stat_features = np.array([0, 0, 0, 0])  # Statistical features of empty barcode
            
            dim_features.append(stat_features)
            
            # Concatenating all features of this dimension
            try:
                dim_features_flat = np.concatenate([arr.flatten() for arr in dim_features])
                # 
                if len(dim_features_flat) != dim_feature_length:
                    logger.warning(f"{protein['id']} dim{dim}Feature length inconsistency: {len(dim_features_flat)} != {dim_feature_length}")
                    # Padding or truncating to correct length
                    if len(dim_features_flat) > dim_feature_length:
                        dim_features_flat = dim_features_flat[:dim_feature_length]
                    else:
                        padding = np.zeros(dim_feature_length - len(dim_features_flat))
                        dim_features_flat = np.concatenate([dim_features_flat, padding])
            except Exception as e:
                logger.error(f"{protein['id']} dim{dim}Feature concatenation failed: {e}")
                dim_features_flat = np.zeros(dim_feature_length)
            
            all_features.append(dim_features_flat)
        
        # Concatenating features of all dimensions
        try:
            protein['tda_vector'] = np.concatenate(all_features)
            # Total feature length
            if len(protein['tda_vector']) != total_feature_length:
                logger.warning(f"{protein['id']}Total feature length inconsistency: {len(protein['tda_vector'])} != {total_feature_length}")
                # Padding or truncating to correct length
                if len(protein['tda_vector']) > total_feature_length:
                    protein['tda_vector'] = protein['tda_vector'][:total_feature_length]
                else:
                    padding = np.zeros(total_feature_length - len(protein['tda_vector']))
                    protein['tda_vector'] = np.concatenate([protein['tda_vector'], padding])
        except Exception as e:
            logger.error(f"{protein['id']}Total feature concatenation failed: {e}")
            protein['tda_vector'] = np.zeros(total_feature_length)
    
    # Verifying all feature vector lengths are consistent
    feature_lengths = [len(p['tda_vector']) for p in data if 'tda_vector' in p]
    if feature_lengths:
        unique_lengths = set(feature_lengths)
        if len(unique_lengths) > 1:
            logger.warning(f"Found feature vectors of different lengths: {unique_lengths}")
            # Unifying length to maximum length
            max_length = max(feature_lengths)
            for p in data:
                if 'tda_vector' in p:
                    if len(p['tda_vector']) < max_length:
                        padding = np.zeros(max_length - len(p['tda_vector']))
                        p['tda_vector'] = np.concatenate([p['tda_vector'], padding])
                    elif len(p['tda_vector']) > max_length:
                        p['tda_vector'] = p['tda_vector'][:max_length]
        logger.info(f"Feature vector length: all proteins unified to {max(feature_lengths) if feature_lengths else 0}")
    else:
        logger.warning("No feature vectors generated")
    
    return data

def process_coordinates(data, config):
    """Processing coordinates: extracting Cα atoms, unifying length, PCA compression"""
    logger.info("Starting coordinate data processing...")
    
    # 1. Extracting Cα atoms
    for protein in data:
        # Ensure coordinates are 2D array
        coords = protein['coords']
        if coords.ndim == 1:
            # , (1, 3)
            coords = coords.reshape(1, -1)
            logger.warning(f"{protein['id']}coordinate is 1D array, converted to 2D")
            protein['coords'] = coords
        
        # Assume every 4th atom is CA
        if len(coords) > 0:
            # Ensure there are enough atoms
            if len(coords) >= 4:
                protein['ca_coords'] = coords[1::4]
            else:
                # ,
                protein['ca_coords'] = coords
                logger.warning(f"{protein['id']}Insufficient number of atoms, using all atoms as Cα")
        else:
            protein['ca_coords'] = np.zeros((config['max_ca_atoms'], 3))
            logger.warning(f"{protein['id']}coordinate is empty, using zero padding")
    
    # 2. Collecting all coordinates - only collecting valid coordinate arrays
    all_coords = []
    for protein in data:
        if 'ca_coords' in protein:
            coords = protein['ca_coords']
            #  (N, 3)
            if coords.size > 0 and coords.ndim == 2 and coords.shape[1] == 3:
                all_coords.append(coords)
            else:
                logger.warning(f"{protein['id']}Cα coordinate shape is invalid: {coords.shape},Skipping global PCA")
    
    if not all_coords:
        logger.error("No valid coordinate data")
        return data, 0
    
    # 3. Global PCA - using only valid coordinates
    all_coords_concat = np.vstack(all_coords)
    pca = PCA(n_components=config['pca_variance'])
    pca.fit(all_coords_concat)
    k = pca.n_components_
    logger.info(f"Global PCA retains{k}principal components, explained variance{pca.explained_variance_ratio_.sum():.2%}")
    
    # 4. Unified length processing
    for protein in tqdm(data, desc=""):
        if 'ca_coords' not in protein:
            continue
            
        coords = protein['ca_coords']
        
        # 
        if coords.size == 0:
            logger.warning(f"{protein['id']}Cα coordinate is empty, using zero padding")
            coords = np.zeros((config['max_ca_atoms'], 3))
            protein['ca_coords'] = coords
        
        # Ensure coordinates are 2D array
        if coords.ndim == 1:
            # ,Reshape to
            if coords.size % 3 == 0:
                coords = coords.reshape(-1, 3)
                logger.warning(f"{protein['id']}Cα coordinate reshaped to 2D")
            else:
                logger.error(f"{protein['id']}Cα coordinate cannot be reshaped: {coords.shape}")
                coords = np.zeros((config['max_ca_atoms'], 3))
                protein['ca_coords'] = coords
        
        # Unifying length
        if len(coords) > config['max_ca_atoms']:
            # Importance sampling (keep more points in central region)
            if len(coords) > 0:
                center = np.mean(coords, axis=0)
                dists = np.linalg.norm(coords - center, axis=1)
                weights = 1 / (dists + 1)
                indices = np.random.choice(len(coords), config['max_ca_atoms'], 
                                          p=weights/weights.sum(), replace=False)
                coords = coords[indices]
        elif len(coords) < config['max_ca_atoms']:
            # Curvature-based interpolation (simplified version)
            if len(coords) > 0:
                last_coord = coords[-1]
                padding = np.tile(last_coord, (config['max_ca_atoms'] - len(coords), 1))
            else:
                padding = np.zeros((config['max_ca_atoms'] - len(coords), 3))
            coords = np.vstack([coords, padding])
        
        # Ensure coordinate shape is correct
        if coords.shape != (config['max_ca_atoms'], 3):
            logger.warning(f"Coordinate shape is incorrect: {coords.shape},Reshape to({config['max_ca_atoms']}, 3)")
            try:
                # Attempting to reshape to correct shape
                coords = coords[:config['max_ca_atoms'], :3]
                if coords.shape[0] < config['max_ca_atoms']:
                    padding = np.zeros((config['max_ca_atoms'] - coords.shape[0], 3))
                    coords = np.vstack([coords, padding])
            except Exception as e:
                logger.error(f"Coordinate reshaping failed: {e}")
                coords = np.zeros((config['max_ca_atoms'], 3))
            protein['ca_coords'] = coords
        
        # PCA transformation
        pca_coords = pca.transform(coords)
        protein['pca_coords'] = pca_coords.flatten()
        protein['pca_object'] = pca
    
    return data, config['max_ca_atoms'] * k

# =====================
# Model building
# =====================

class TopologyRegularizer(tf.keras.layers.Layer):
    """Topology regularization layer"""
    def __init__(self, factor=0.01, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        
    def call(self, inputs):
        # Computing regularization term: mean(square(inputs)) * factor
        reg = tf.reduce_mean(tf.square(inputs)) * self.factor
        
        # Adding as model loss
        self.add_loss(reg)
        
        # Adding as monitoring metric
        self.add_metric(reg, name='topology_reg', aggregation='mean')
        
        return inputs  # Return input unchanged for use by subsequent layers
    
    def get_config(self):
        config = super().get_config()
        config.update({'factor': self.factor})
        return config

def build_topology_aware_model(input_dim, output_dim, latent_dim=256):
    """Building topology-aware autoencoder model"""
    logger.info(f"Building model: input dimension={input_dim}, output dimension={output_dim}, latent dimension={latent_dim}")
    
    # 1. Input layer
    inputs = Input(shape=(input_dim,))
    
    # 2. Encoder part
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    encoded = Dense(latent_dim, activation='relu', name="latent_topology")(x)
    
    # 3. Using custom layer to add topology constraint
    class TopologyConstraint(tf.keras.layers.Layer):
        def __init__(self, factor=0.01, **kwargs):
            super().__init__(**kwargs)
            self.factor = factor
            
        def call(self, inputs):
            # Computing regularization term
            reg = self.factor * tf.reduce_mean(tf.square(inputs))
            
            # Adding as model loss
            self.add_loss(reg)
            
            return inputs  # Return input unchanged
        
        # Fixing output shape inference issue
        def compute_output_spec(self, inputs):
            # Return KerasTensor instead of TensorSpec
            return tf.keras.KerasTensor(
                shape=inputs.shape, 
                dtype=inputs.dtype,
                name=self.name + '_output'
            )
        
        # Compatible with legacy API
        def compute_output_shape(self, input_shape):
            return input_shape
        
        def get_config(self):
            config = super().get_config()
            config.update({'factor': self.factor})
            return config
    
    # Using custom layer
    encoded = TopologyConstraint(factor=0.01, name="topo_constraint")(encoded)
    
    # 4. Decoder part
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(encoded)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
    decoded = Dense(output_dim, activation='linear')(x)
    
    # 5. Creating autoencoder model
    autoencoder = Model(inputs, decoded)
    
    # 6. Compiling model
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    # Printing model summary
    logger.info("Model architecture summary:")
    autoencoder.summary(print_fn=lambda x: logger.info(x))
    
    logger.info("Model construction completed")
    return autoencoder

# =====================
# Evaluation functions
# =====================

def calculate_rmsd(coord1, coord2):
    """Computing RMSD between two sets of coordinates"""
    if len(coord1) != len(coord2):
        min_len = min(len(coord1), len(coord2))
        coord1 = coord1[:min_len]
        coord2 = coord2[:min_len]
    
    diff = coord1 - coord2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def calculate_tm_score(coord1, coord2):
    """Computing TM-score"""
    if len(coord1) != len(coord2):
        min_len = min(len(coord1), len(coord2))
        coord1 = coord1[:min_len]
        coord2 = coord2[:min_len]
    
    d0 = 1.24 * (len(coord1) - 15) ** (1/3) - 1.8  # Empirical formula
    dists = np.linalg.norm(coord1 - coord2, axis=1)
    tm = np.sum(1 / (1 + (dists/d0)**2)) / len(coord1)
    return tm

def compute_barcode(coords, dim=1):
    """Computing barcode for given coordinates (specified dimension)"""
    try:
        # Creating Rips complex
        rips_complex = gudhi.RipsComplex(points=coords)
        # Creating simplex complex
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=dim+1)
        # Computing persistent homology
        persistence = simplex_tree.persistence()
        
        # Extracting barcode of specified dimension
        barcode = []
        for interval in persistence:
            if interval[0] == dim:
                barcode.append(interval[1])
        
        return np.array(barcode)
    except Exception as e:
        logger.error(f"Failed to compute barcode: {e}")
        return np.empty((0, 2))

def evaluate_reconstruction(protein_data, reconstructed_coords):
    """Evaluating reconstruction quality: geometric and topological metrics"""
    logger.info("Starting reconstruction quality evaluation...")
    
    # Attempting to import Wasserstein distance function
    try:
        from persim import wasserstein
        has_wasserstein = True
        logger.info("Successfully imported persim.wasserstein module")
    except ImportError:
        logger.warning("Failed to import persim module, skipping Wasserstein distance calculation")
        has_wasserstein = False
    
    results = {
        'geometric': {'RMSD': [], 'TM-score': []},
        'topological': {'Wasserstein_dim1': []},
        'per_protein': []
    }
    
    for i, protein in enumerate(tqdm(protein_data, desc="Evaluation")):
        # Geometric evaluation
        orig_coords = protein['ca_coords']
        rec_coords = reconstructed_coords[i].reshape(-1, 3)
        
        # Computing RMSD
        if len(orig_coords) != len(rec_coords):
            min_len = min(len(orig_coords), len(rec_coords))
            orig_coords = orig_coords[:min_len]
            rec_coords = rec_coords[:min_len]
        
        diff = orig_coords - rec_coords
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        
        # Computing TM-score
        tm_score = calculate_tm_score(orig_coords, rec_coords)
        
        # Topological evaluation (computed only for original structures)
        w_dist = None
        if "mirrored" not in protein['id'] and has_wasserstein:
            try:
                orig_barcode_dim1 = compute_barcode(orig_coords, dim=1)
                rec_barcode_dim1 = compute_barcode(rec_coords, dim=1)
                
                if len(orig_barcode_dim1) > 0 and len(rec_barcode_dim1) > 0:
                    # persimWasserstein
                    w_dist = wasserstein(orig_barcode_dim1, rec_barcode_dim1)
                else:
                    w_dist = 1.0  # 
                    logger.warning(f"{protein['id']}Barcode is empty, using default Wasserstein distance 1.0")
            except Exception as e:
                logger.error(f"{protein['id']}Topological evaluation failed: {e}")
                w_dist = 1.0
        
        # Recording results
        results['geometric']['RMSD'].append(rmsd)
        results['geometric']['TM-score'].append(tm_score)
        if w_dist is not None:
            results['topological']['Wasserstein_dim1'].append(w_dist)
        
        results['per_protein'].append({
            'id': protein['id'],
            'RMSD': rmsd,
            'TM-score': tm_score,
            'Wasserstein': w_dist
        })
    
    # Computing statistics
    results['mean_RMSD'] = np.mean(results['geometric']['RMSD'])
    results['mean_TM'] = np.mean(results['geometric']['TM-score'])
    if results['topological']['Wasserstein_dim1']:
        results['mean_Wasserstein'] = np.mean(results['topological']['Wasserstein_dim1'])
    else:
        results['mean_Wasserstein'] = None
    
    logger.info(f"Evaluation completed: Mean RMSD={results['mean_RMSD']:.4f}Å, Mean TM-score={results['mean_TM']:.4f}")
    return results

# =====================
# Visualization functions
# =====================

def plot_3d_comparison(orig_coords, rec_coords, title, save_path):
    """Plotting 3D structure comparison"""
    fig = plt.figure(figsize=(14, 6))
    
    # Original Structure
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(orig_coords[:,0], orig_coords[:,1], orig_coords[:,2], 
               c='blue', s=20, alpha=0.6, label='Original')
    ax1.set_title(f'Original Structure: {title}')
    
    # Reconstructed Structure
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(rec_coords[:,0], rec_coords[:,1], rec_coords[:,2], 
               c='red', s=20, alpha=0.6, label='Reconstructed')
    ax2.set_title(f'Reconstructed Structure')
    
    # Unified settings
    for ax in [ax1, ax2]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Structure comparison map saved to: {save_path}")

def plot_topology_geometry_correlation(results, save_path):
    """Plotting topology-geometry correlation"""
    # Extracting valid data (excluding mirrored structures)
    valid_results = [r for r in results['per_protein'] 
                   if r['Wasserstein'] is not None and "mirrored" not in r['id']]
    
    # Adding detailed logging information
    total_proteins = len(results['per_protein'])
    mirrored_count = len([r for r in results['per_protein'] if "mirrored" in r['id']])
    wasserstein_none_count = len([r for r in results['per_protein'] if r['Wasserstein'] is None])
    valid_count = len(valid_results)
    
    logger.info(f"Topology-geometry correlation analysis statistics:")
    logger.info(f"  Total number of proteins: {total_proteins}")
    logger.info(f"  Number of mirrored structures: {mirrored_count}")
    logger.info(f"  Number of None Wasserstein distances: {wasserstein_none_count}")
    logger.info(f"  Number of valid data points: {valid_count}")
    
    if not valid_results:
        logger.warning(f"No valid data for topology-geometry correlation analysis, reasons:")
        if valid_count == 0:
            logger.warning("  - Wasserstein distances for all proteins are None")
        if mirrored_count == total_proteins:
            logger.warning("  - All proteins are mirrored structures")
        if wasserstein_none_count == total_proteins - mirrored_count:
            logger.warning("  - Wasserstein distances for all non-mirrored proteins are None")
        return
    
    # Continuing to plot correlation graph...
    rmsd_values = [r['RMSD'] for r in valid_results]
    wasserstein_values = [r['Wasserstein'] for r in valid_results]
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(
        rmsd_values, 
        wasserstein_values, 
        alpha=0.6,
        edgecolor='k',       
        linewidths=0.5       
    )
    # **********************************************
    
    # Adding trend line
    if len(rmsd_values) > 1:
        z = np.polyfit(rmsd_values, wasserstein_values, 1)
        p = np.poly1d(z)
        plt.plot(rmsd_values, p(rmsd_values), "r--")
        correlation = np.corrcoef(rmsd_values, wasserstein_values)[0, 1]
        plt.title(f"Topology-Geometry Correlation (r = {correlation:.2f})")
    else:
        plt.title("Topology-Geometry Correlation")
    
    plt.xlabel("RMSD (Å)")
    plt.ylabel("Wasserstein Distance")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Topology-geometry correlation map saved to: {save_path}")

# =====================
# Batch processing functions
# =====================

def process_data_in_chunks(config, total_proteins):
    """Dividing large dataset into multiple batches for processing"""
    # Creating batch output directory
    batch_output_dir = os.path.join(config['output_dir'], "batch_results")
    os.makedirs(batch_output_dir, exist_ok=True)
    
    # Computing number of batches
    num_chunks = (total_proteins + config['chunk_size'] - 1) // config['chunk_size']
    logger.info(f"Total number of proteins: {total_proteins}, Divided into {num_chunks} batches, each batch {config['chunk_size']} proteins")
    
    # Processing each batch
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * config['chunk_size']
        end_idx = min((chunk_idx + 1) * config['chunk_size'], total_proteins)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing batch {chunk_idx+1}/{num_chunks} (index {start_idx}-{end_idx-1})")
        logger.info(f"{'='*50}")
        
        # 
        process_single_batch(config, start_idx, end_idx, chunk_idx, batch_output_dir)
    
    # Combining all batch results
    combine_batch_results(config, batch_output_dir, num_chunks)

def process_single_batch(config, start_idx, end_idx, chunk_idx, batch_output_dir):
    """Processing single batch of data"""
    # Creating independent output directory for current batch
    chunk_output_dir = os.path.join(batch_output_dir, f"batch_{chunk_idx}")
    os.makedirs(chunk_output_dir, exist_ok=True)
    
    # Loading data
    protein_data = load_and_process_data(config, start_idx, end_idx)
    if not protein_data:
        logger.error(f" {chunk_idx} No valid data, skipping")
        return
    
    # Feature engineering
    protein_data = vectorize_tda_features(protein_data, config)
    
    # Checking feature vector length
    feature_lengths = [len(p['tda_vector']) for p in protein_data if 'tda_vector' in p]
    if feature_lengths:
        max_length = max(feature_lengths)
        for p in protein_data:
            if 'tda_vector' in p and len(p['tda_vector']) != max_length:
                padding = np.zeros(max_length - len(p['tda_vector']))
                p['tda_vector'] = np.concatenate([p['tda_vector'], padding])
    
    # Processing coordinates
    protein_data, output_dim = process_coordinates(protein_data, config)
    
    # Preparing model input
    X = np.array([p['tda_vector'] for p in protein_data])
    y = np.array([p['pca_coords'] for p in protein_data])
    
    # Dividing dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['random_state']
    )
    
    # Building and training model
    model = build_topology_aware_model(X_train.shape[1], y_train.shape[1], 
                                      latent_dim=config['latent_dim'])
    
    logger.info("Starting model training...")
    
    # 
    history = model.fit(
        X_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=0.1,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=10, 
                restore_best_weights=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(chunk_output_dir, f'best_model_batch_{chunk_idx}.keras'),
                save_best_only=True,
                monitor='val_loss'
            )
        ]
    )
    
    # Evaluation
    logger.info("Evaluating model...")
    all_pred = model.predict(X)
    results = evaluate_reconstruction(protein_data, all_pred)
    
    # 
    save_batch_results(chunk_output_dir, chunk_idx, results, history, protein_data, all_pred)

def save_batch_results(output_dir, batch_idx, results, history, protein_data, all_pred):
    """Save intermediate results for current batch"""
    # Evaluation
    results_df = pd.DataFrame(results['per_protein'])
    results_path = os.path.join(output_dir, f'reconstruction_metrics_batch_{batch_idx}.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Evaluation results saved to: {results_path}")
    
    # Saving training history
    history_path = os.path.join(output_dir, f'training_history_batch_{batch_idx}.pkl')
    joblib.dump(history.history, history_path)
    
    # Saving protein data and prediction results
    data_path = os.path.join(output_dir, f'protein_data_batch_{batch_idx}.pkl')
    joblib.dump({
        'protein_data': protein_data,
        'all_pred': all_pred
    }, data_path)
    
    # Plotting charts for current batch
    plot_batch_visualizations(output_dir, batch_idx, results, history, protein_data, all_pred)
    
    logger.info(f" {batch_idx} Intermediate results saved")

def plot_batch_visualizations(output_dir, batch_idx, results, history, protein_data, all_pred):
    """Generating charts for current batch"""
    # Training history visualization
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Batch {batch_idx}: Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    save_path = os.path.join(output_dir, f'training_history_batch_{batch_idx}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

    # Topology-geometry correlation
    corr_path = os.path.join(output_dir, f'topology_geometry_correlation_batch_{batch_idx}.png')
    plot_topology_geometry_correlation(results, corr_path)
    
    # Randomly selecting 2 non-mirrored proteins for 3D visualization
    non_mirror = [p for p in protein_data if "mirrored" not in p['id']]
    if non_mirror:
        sample_indices = np.random.choice(len(non_mirror), min(2, len(non_mirror)), replace=False)
        
        for idx in sample_indices:
            protein = non_mirror[idx]
            orig_coords = protein['ca_coords']
            
            # Getting predicted coordinates
            protein_index = protein_data.index(protein)
            pred_flat = all_pred[protein_index]
            pca = protein['pca_object']
            rec_coords = pca.inverse_transform(pred_flat.reshape(-1, pca.n_components_))
            
            # Plotting structure comparison
            plot_path = os.path.join(output_dir, f"structure_comparison_batch_{batch_idx}_{protein['id']}.png")
            plot_3d_comparison(orig_coords, rec_coords, protein['id'], plot_path)

def combine_batch_results(config, batch_output_dir, num_batches):
    """Combining all batch results and generating final charts"""
    logger.info("\n" + "="*50)
    logger.info("Starting to combine all batch results")
    logger.info("="*50)
    
    # Creating final results directory
    final_output_dir = os.path.join(config['output_dir']+'/batch{}-{}'.format(start_idx, end_idx), "final_results")
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Combining all evaluation results
    all_results = []
    for i in range(num_batches):
        batch_dir = os.path.join(batch_output_dir, f"batch_{i}")
        results_file = os.path.join(batch_dir, f'reconstruction_metrics_batch_{i}.csv')
        
        if os.path.exists(results_file):
            try:
                batch_df = pd.read_csv(results_file)
                all_results.append(batch_df)
                logger.info(f"Loading batch {i} results successful, contains {len(batch_df)} proteins")
            except Exception as e:
                logger.error(f"Loading batch {i} results failed: {e}")
    
    if not all_results:
        logger.error("No batch results found")
        return
    
    full_results_df = pd.concat(all_results, ignore_index=True)
    full_results_path = os.path.join(final_output_dir, 'full_reconstruction_metrics.csv')
    full_results_df.to_csv(full_results_path, index=False)
    logger.info(f"Combined evaluation results saved to: {full_results_path}")
    
    # Generating final correlation chart
    plot_full_correlation(full_results_df, final_output_dir)
    
    # Generating final training history chart
    plot_full_training_history(batch_output_dir, num_batches, final_output_dir)
    
    # Generating final structure comparison charts (sampling from each batch)
    plot_final_structure_comparisons(batch_output_dir, num_batches, final_output_dir, config)
    
    logger.info("All batch results successfully combined!")

def plot_full_correlation(full_results_df, output_dir):
    """Plotting topology-geometry correlation for entire dataset"""
    # Extracting valid data (excluding mirrored structures)
    valid_results = full_results_df[
        (full_results_df['Wasserstein'].notnull()) & 
        (~full_results_df['id'].str.contains('mirrored'))
    ]
    
    if valid_results.empty:
        logger.warning("No valid data for topology-geometry correlation analysis")
        return
    
    rmsd_values = valid_results['RMSD'].values
    wasserstein_values = valid_results['Wasserstein'].values
    
    plt.figure(figsize=(10, 6))
    plt.scatter(rmsd_values, wasserstein_values, edgecolor='black', linewidth=0.3, alpha=0.6)
    
    # Adding trend line
    if len(rmsd_values) > 1:
        z = np.polyfit(rmsd_values, wasserstein_values, 1)
        p = np.poly1d(z)
        plt.plot(rmsd_values, p(rmsd_values), "r--")
        correlation = np.corrcoef(rmsd_values, wasserstein_values)[0, 1]
        plt.title(f"Full Dataset: Topology-Geometry Correlation (r = {correlation:.2f})")
    else:
        plt.title("Full Dataset: Topology-Geometry Correlation")
    
    plt.xlabel("RMSD (Å)")
    plt.ylabel("Wasserstein Distance")
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, 'full_topology_geometry_correlation.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Full dataset topology-geometry correlation map saved to: {save_path}")
    
    # Adding statistical information
    stats = {
        'mean_RMSD': np.mean(rmsd_values),
        'std_RMSD': np.std(rmsd_values),
        'mean_Wasserstein': np.mean(wasserstein_values),
        'std_Wasserstein': np.std(wasserstein_values),
        'correlation': correlation if len(rmsd_values) > 1 else None,
        'num_proteins': len(valid_results)
    }
    
    stats_path = os.path.join(output_dir, 'correlation_stats.txt')
    with open(stats_path, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Correlation statistical information saved to: {stats_path}")

def plot_full_training_history(batch_output_dir, num_batches, output_dir):
    """Plotting training history for all batches"""
    plt.figure(figsize=(12, 8))
    
    for i in range(num_batches):
        batch_dir = os.path.join(batch_output_dir, f"batch_{i}")
        history_file = os.path.join(batch_dir, f'training_history_batch_{i}.pkl')
        
        if os.path.exists(history_file):
            try:
                history = joblib.load(history_file)
                min_epochs = min(len(history['loss']), len(history['val_loss']))
                plt.plot(history['loss'][:min_epochs], label=f'Batch {i} Training Loss', alpha=0.7)
                plt.plot(history['val_loss'][:min_epochs], label=f'Batch {i} Validation Loss', linestyle='--', alpha=0.7)
            except Exception as e:
                logger.error(f"Loading batch {i} failed: {e}")
    
    plt.title('Combined Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, 'combined_training_history.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Combined training history map saved to: {save_path}")

def plot_final_structure_comparisons(batch_output_dir, num_batches, output_dir, config):
    """Sampling and generating structure comparison charts from each batch"""
    # Randomly selecting a proteinrows
    for i in range(num_batches):
        batch_dir = os.path.join(batch_output_dir, f"batch_{i}")
        data_file = os.path.join(batch_dir, f'protein_data_batch_{i}.pkl')
        
        if os.path.exists(data_file):
            try:
                batch_data = joblib.load(data_file)
                protein_data = batch_data['protein_data']
                all_pred = batch_data['all_pred']
                
                # Selecting non-mirrored proteins
                non_mirror = [p for p in protein_data if "mirrored" not in p['id']]
                if non_mirror:
                    # Randomly selecting a protein
                    protein = np.random.choice(non_mirror)
                    
                    # Getting original coordinates
                    orig_coords = protein['ca_coords']
                    
                    # Getting predicted coordinates
                    protein_index = protein_data.index(protein)
                    pred_flat = all_pred[protein_index]
                    pca = protein['pca_object']
                    rec_coords = pca.inverse_transform(pred_flat.reshape(-1, pca.n_components_))
                    
                    # Plotting structure comparison
                    plot_path = os.path.join(output_dir, f"structure_comparison_batch_{i}_{protein['id']}.png")
                    plot_3d_comparison(orig_coords, rec_coords, f"Batch {i}: {protein['id']}", plot_path)
            except Exception as e:
                logger.error(f"Loading batch {i} Protein data failed: {e}")

    # Saving statistical information
    stats = {
        'mean_RMSD': np.mean(rmsd_values),
        'median_RMSD': np.median(rmsd_values),
        'std_RMSD': np.std(rmsd_values),
        'mean_TM-score': np.mean(tm_scores),
        'median_TM-score': np.median(tm_scores),
        'mean_Wasserstein': np.mean(wasserstein_values),
        'median_Wasserstein': np.median(wasserstein_values)
    }
    
    stats_path = os.path.join(error_dir, 'error_stats.txt')
    with open(stats_path, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Reconstruction error analysis results saved to: {error_dir}")

def plot_structure_comparison(original_coords, mirror_coords, original_diagram, mirror_diagram, protein_id):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from persim import plot_diagrams

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(original_coords[:, 0], original_coords[:, 1], original_coords[:, 2], color='blue')
    ax1.set_title(f'Original Structure: {protein_id}')
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot(mirror_coords[:, 0], mirror_coords[:, 1], mirror_coords[:, 2], color='red')
    ax2.set_title(f'Mirror Structure: {protein_id}')
    ax2.axis('off')

    ax3 = fig.add_subplot(2, 2, 3)
    plot_diagrams(original_diagram, ax=ax3)
    ax3.set_title('Original Barcode')

    ax4 = fig.add_subplot(2, 2, 4)
    plot_diagrams(mirror_diagram, ax=ax4)
    ax4.set_title('Mirror Barcode')

    plt.tight_layout()
    plt.savefig(f'figures/{protein_id}_structure_barcode_comparison.png', dpi=300)
    plt.close()

# =====================
# Main function
# =====================

def main(config, start_idx=0, end_idx=None, batch_mode=False):
    """Main workflow"""
    # 1. Preparing output directory
    os.makedirs(config['output_dir']+'/batch{}-{}'.format(start_idx, end_idx), exist_ok=True)
    logger.info(f"Output directory: {config['output_dir']}")
    
    if batch_mode:
        # Batch processing mode
        # Getting total number of proteins
        try:
            barcode_df = pd.read_csv(config['tda_csv_path'])
            total_proteins = len(barcode_df['ID'].unique())
            logger.info(f"Total number of proteins: {total_proteins}")
            
            # 
            process_data_in_chunks(config, total_proteins)
        except Exception as e:
            logger.error(f"Failed to load TDA data: {e}")
            return
    else:
        # Single batch processing mode (original functionality)
        # 2. Loading data
        protein_data = load_and_process_data(config, start_idx, end_idx)
        if not protein_data:
            logger.error("No valid data, program terminated")
            return
        
        # 3. Feature engineering
        protein_data = vectorize_tda_features(protein_data, config)
        
        # Checking if feature vector lengths are consistent
        feature_lengths = [len(p['tda_vector']) for p in protein_data]
        if len(set(feature_lengths)) > 1:
            logger.error(f"Error: Feature vector lengths inconsistent: {set(feature_lengths)}")
            # Unifying length to maximum length
            max_length = max(feature_lengths)
            for p in protein_data:
                if len(p['tda_vector']) < max_length:
                    padding = np.zeros(max_length - len(p['tda_vector']))
                    p['tda_vector'] = np.concatenate([p['tda_vector'], padding])
                elif len(p['tda_vector']) > max_length:
                    p['tda_vector'] = p['tda_vector'][:max_length]
        
        protein_data, output_dim = process_coordinates(protein_data, config)    
        
        # 4. Preparing model input
        X = np.array([p['tda_vector'] for p in protein_data])
        y = np.array([p['pca_coords'] for p in protein_data])
        
        # 5. Dividing dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config['test_size'], random_state=config['random_state']
        )
        
        # 6. Building and training model
        model = build_topology_aware_model(X_train.shape[1], y_train.shape[1], 
                                          latent_dim=config['latent_dim'])
        
        logger.info("Starting model training...")
        
        # Creating custom callback to monitor topology regularization
        class TopologyMonitor(tf.keras.callbacks.Callback):
            def __init__(self, model, X_train):
                super().__init__()
                self.custom_model = model
                self.latent_model = tf.keras.Model(
                    inputs=model.input,
                    outputs=model.get_layer('latent_topology').output
                )
                self.X_train = X_train
            
            def on_epoch_end(self, epoch, logs=None):
                latent_output = self.latent_model.predict(self.X_train, verbose=0)
                topo_reg = 0.01 * np.mean(np.square(latent_output))
                logs['topology_reg'] = topo_reg
                logger.info(f"Epoch {epoch+1}: Topology regularization = {topo_reg:.6f}")

        # 
        history = model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_split=0.1,
            verbose=1,
            callbacks=[
                TopologyMonitor(model, X_train),
                tf.keras.callbacks.EarlyStopping(
                    patience=10, 
                    restore_best_weights=True,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(config['output_dir']+'/batch{}-{}'.format(start_idx, end_idx),'best_model_batch{}-{}.keras'.format(start_idx, end_idx)),
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]
        )
        
        # 7. Evaluation
        logger.info("Evaluating model...")
        all_pred = model.predict(X)
        results = evaluate_reconstruction(protein_data, all_pred)
        
        # 8. Saving results
        results_df = pd.DataFrame(results['per_protein'])
        results_path = os.path.join(config['output_dir']+'/batch{}-{}'.format(start_idx, end_idx), 'reconstruction_metrics_batch{}-{}.csv'.format(start_idx, end_idx))
        results_df.to_csv(results_path, index=False)
        logger.info(f"Evaluation results saved to: {results_path}")
        
        # 9. Key visualization
        logger.info("Generating visualization charts...")
        
        # Randomly selecting 5 non-mirrored proteins for 3D visualization
        non_mirror = [p for p in protein_data if "mirrored" not in p['id']]
        if non_mirror:
            sample_indices = np.random.choice(len(non_mirror), min(5, len(non_mirror)), replace=False)
            
            for idx in sample_indices:
                protein = non_mirror[idx]
                orig_coords = protein['ca_coords']
                
                # Getting predicted coordinates
                protein_index = protein_data.index(protein)
                pred_flat = all_pred[protein_index]
                pca = protein['pca_object']
                rec_coords = pca.inverse_transform(pred_flat.reshape(-1, pca.n_components_))
                
                # Plotting structure comparison
                plot_path = os.path.join(config['output_dir']+'/batch{}-{}'.format(start_idx, end_idx), f"structure_comparison_{protein['id']}.png")
                plot_3d_comparison(orig_coords, rec_coords, protein['id'], plot_path)
        
        # Topology-geometry correlation
        corr_path = os.path.join(config['output_dir']+'/batch{}-{}'.format(start_idx, end_idx), 'topology_geometry_correlation_batch{}-{}.png'.format(start_idx, end_idx))
        plot_topology_geometry_correlation(results, corr_path)

        # inEvaluation
        logger.info("Starting execution of new analysis module...")
        
        # Getting result DataFrame
        results_df = pd.DataFrame(results['per_protein'])
          
        # Training history visualization
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        save_path = os.path.join(config['output_dir']+'/batch{}-{}'.format(start_idx, end_idx), 'training_history_batch{}-{}.png'.format(start_idx, end_idx))
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    logger.info("All processing completed!")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

# =====================
# Command-line argument parsing
# =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Protein structure reconstruction experiment: Phase 1')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting index (default 0)')
    parser.add_argument('--end_idx', type=int, default=69919, help='End index (default None=read to end of file)')
    parser.add_argument('--batch_mode', action='store_true', help='Enable batch processing mode')
    parser.add_argument('--max_ca_atoms', type=int, default=200, help='Maximum Cα atom count')
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent space dimension')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--pca_variance', type=float, default=0.95, help='PCA retained variance ratio')
    parser.add_argument('--chunk_size', type=int, default=2000, help='Number of proteins to process per batch')
    args = parser.parse_args()
    
    # Updating configuration
    config = DEFAULT_CONFIG.copy()
    config['max_ca_atoms'] = args.max_ca_atoms
    config['latent_dim'] = args.latent_dim
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['pca_variance'] = args.pca_variance
    config['chunk_size'] = args.chunk_size
    
    # Run main function
    main(config, args.start_idx, args.end_idx, batch_mode=args.batch_mode)


# How to use: python step4.1_F3DR_high_dim_projection_regression.py --start_idx 0 --end_idx 69919
