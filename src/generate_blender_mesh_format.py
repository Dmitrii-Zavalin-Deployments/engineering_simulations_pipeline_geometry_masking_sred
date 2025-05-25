import numpy as np
import json
import os
import openvdb  # Handles volumetric data processing
import alembic  # Exports to Alembic mesh format
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes


def load_simulation_data(workspace):
    """ Load relevant fluid simulation data. """
    data_dir = os.path.join(workspace, "data/testing-input-output")

    # Load metadata & simulation files
    with open(os.path.join(data_dir, "grid_metadata.json"), "r") as f:
        grid_metadata = json.load(f)

    nodes_coords = np.load(os.path.join(data_dir, "nodes_coords.npy"))
    velocity_history = np.load(os.path.join(data_dir, "velocity_history.npy"))
    pressure_history = np.load(os.path.join(data_dir, "pressure_history.npy"))
    turbulence_history = np.load(os.path.join(data_dir, "turbulence_kinetic_energy_history.npy"))

    return grid_metadata, nodes_coords, velocity_history, pressure_history, turbulence_history


def extract_fluid_surface(pressure_history, velocity_history, turbulence_history):
    """ Identify fluid surface using hybrid approach: pressure + velocity + turbulence. """

    # Compute smoothed pressure gradient
    pressure_grad = np.gradient(pressure_history, axis=-1)
    pressure_grad_smoothed = gaussian_filter(pressure_grad, sigma=1)

    # Detect surface areas with significant pressure drop
    surface_mask = pressure_grad_smoothed > np.mean(pressure_grad_smoothed) * 0.5

    # Refine using velocity: areas with high motion contribute to fluid surface formation
    velocity_magnitude = np.linalg.norm(velocity_history, axis=-1)
    surface_mask = np.logical_and(surface_mask, velocity_magnitude > np.mean(velocity_magnitude))

    # Apply turbulence refinement: fluid surface features depend on localized turbulence
    turbulence_smoothed = gaussian_filter(turbulence_history, sigma=1)
    surface_mask = np.logical_and(surface_mask, turbulence_smoothed > np.mean(turbulence_smoothed) * 0.2)

    return surface_mask


def generate_mesh(surface_mask, nodes_coords):
    """ Convert extracted surface data into a triangular mesh using Marching Cubes. """

    # Transform surface mask into a volumetric representation
    surface_field = surface_mask.astype(np.float32)

    # Extract surface mesh using Marching Cubes algorithm
    verts, faces, normals, _ = marching_cubes(surface_field, level=0.5)

    # Map vertices to physical space
    verts_transformed = nodes_coords[verts.astype(int)]

    return verts_transformed, faces, normals


def export_to_alembic(verts, faces, output_file):
    """ Convert mesh data to Alembic (.abc) format for animation/rendering. """
    archive = alembic.AbcGeom.OArchive(output_file)
    mesh_obj = alembic.AbcGeom.OPolyMesh(archive.getTop(), "fluid_mesh")

    # Store mesh geometry
    mesh_schema = mesh_obj.getSchema()
    mesh_sample = alembic.AbcGeom.OPolyMeshSchemaSample(verts, faces)
    mesh_schema.set(mesh_sample)

    print(f"✅ Fluid surface mesh exported to {output_file}")


# Main execution
if __name__ == "__main__":
    workspace = os.getenv("GITHUB_WORKSPACE", ".")
    output_file = os.path.join(workspace, "data/testing-input-output/fluid_mesh.abc")

    # Load simulation data
    grid_metadata, nodes_coords, velocity_history, pressure_history, turbulence_history = load_simulation_data(workspace)

    # Extract fluid surface using pressure + velocity + turbulence hybrid method
    surface_mask = extract_fluid_surface(pressure_history, velocity_history, turbulence_history)

    # Generate mesh from extracted surface data
    verts, faces, normals = generate_mesh(surface_mask, nodes_coords)

    # Export mesh in Alembic format
    export_to_alembic(verts, faces, output_file)
