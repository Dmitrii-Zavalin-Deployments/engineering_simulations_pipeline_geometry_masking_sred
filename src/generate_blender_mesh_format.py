import numpy as np
import json
import os
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

    # Ensure fluid regions are properly detected
    threshold_value = np.percentile(pressure_grad_smoothed, 50)  # Use the top 50% of variations
    surface_mask = pressure_grad_smoothed > threshold_value

    # Ensure velocity contributes to surface formation
    velocity_magnitude = np.linalg.norm(velocity_history, axis=-1)
    velocity_threshold = np.percentile(velocity_magnitude, 50)
    surface_mask = np.logical_or(surface_mask, velocity_magnitude > velocity_threshold)

    # Ensure turbulence refines the surface structure
    turbulence_smoothed = gaussian_filter(turbulence_history, sigma=1)
    turbulence_threshold = np.percentile(turbulence_smoothed, 50)
    surface_mask = np.logical_or(surface_mask, turbulence_smoothed > turbulence_threshold)

    return surface_mask


def generate_mesh(surface_mask, nodes_coords):
    """ Convert extracted surface data into a triangular mesh using Marching Cubes. """

    # Convert surface mask to float for processing
    surface_field = surface_mask.astype(np.float32)

    # Debugging: Print shape and value range before attempting Marching Cubes
    print(f"🔍 Debugging: surface_field shape = {surface_field.shape}")
    print(f"🔍 Unique values in surface_field: {np.unique(surface_field)}")
    print(f"🔍 surface_field min = {surface_field.min()}, max = {surface_field.max()}")

    # Ensure the input is a valid 3D array
    if surface_field.ndim == 4:
        print(f"⚠️ Input is 4D. Selecting the first time step...")
        surface_field = surface_field[0]  # Extract single 3D frame

    elif surface_field.ndim > 3:
        print(f"⚠️ Input is higher than 3D. Averaging over extra dimensions...")
        surface_field = np.mean(surface_field, axis=0)

    elif surface_field.ndim < 3:
        raise ValueError(f"❌ Error: Expected 3D input but found {surface_field.ndim}D array.")

    print(f"✅ Fixed shape: surface_field = {surface_field.shape}")

    # Adjust surface level dynamically to prevent selection outside valid range
    min_val, max_val = surface_field.min(), surface_field.max()
    surface_level = max(np.mean(surface_field), min_val + 0.05)

    # Prevent errors when all values are zero
    if min_val == max_val:
        raise RuntimeError("❌ No valid surface detected: All values in surface_field are identical.")

    print(f"✅ Adjusted surface level: {surface_level}")

    # Extract surface mesh using Marching Cubes algorithm
    verts, faces, normals, _ = marching_cubes(surface_field, level=surface_level)

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
