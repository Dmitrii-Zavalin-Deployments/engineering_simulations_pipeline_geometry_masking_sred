import numpy as np
import json
import os
import alembic
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes


def load_simulation_data(workspace):
    """
    Loads relevant fluid simulation data from specified paths.
    """
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
    """
    Identifies the fluid surface by creating a scalar field representing "fluidness."
    Higher values in the field indicate a higher likelihood of being fluid.
    This field is then smoothed for better mesh generation.
    """

    print("\n--- Debugging extract_fluid_surface ---")

    # --- Pre-process data for a single time step if it's 4D ---
    # The generate_mesh function will handle the 4D to 3D conversion,
    # but for individual contributions here, let's work on the first timestep.
    # We should ensure each history array is 3D before calculating contributions.
    if pressure_history.ndim == 4:
        pressure_history_3d = pressure_history[0]
        velocity_history_3d = velocity_history[0]
        turbulence_history_3d = turbulence_history[0]
        print(f"ℹ️ Extracting first time step for surface extraction: {pressure_history_3d.shape}")
    else:
        pressure_history_3d = pressure_history
        velocity_history_3d = velocity_history
        turbulence_history_3d = turbulence_history
        print(f"ℹ️ Input histories are already 3D: {pressure_history_3d.shape}")

    # --- Pressure contribution ---
    print(f"  Pressure (3D) shape: {pressure_history_3d.shape}")
    print(f"  Pressure min: {pressure_history_3d.min()}, max: {pressure_history_3d.max()}")
    print(f"  Pressure unique values: {np.unique(pressure_history_3d)}")

    pressure_contribution = pressure_history_3d
    if pressure_contribution.max() > pressure_contribution.min():
        pressure_contribution = (pressure_contribution - pressure_contribution.min()) / \
                                (pressure_contribution.max() - pressure_contribution.min())
        print(f"  Pressure contribution normalized. Min: {pressure_contribution.min()}, Max: {pressure_contribution.max()}")
    else:
        pressure_contribution = np.zeros_like(pressure_contribution)
        print("  Pressure data is uniform, contribution set to zeros.")

    # --- Velocity magnitude contribution ---
    # Ensure axis=-1 is correct if velocity_history_3d is (D, H, W, 3)
    # If it's (D, H, W) and velocity is a scalar magnitude, then axis=-1 would be wrong for norm.
    # Assuming velocity_history is (time, D, H, W, 3) or (D, H, W, 3)
    # If velocity_history_3d is (D, H, W) of scalar magnitudes, then axis=-1 for norm doesn't make sense.
    # np.linalg.norm works on the last axis for a vector.
    # If your velocity_history_3d is simply (Z, Y, X) of scalar magnitudes, remove axis=-1.
    # Otherwise, it needs to be (Z, Y, X, 3) for vectors.
    if velocity_history_3d.shape[-1] == 3: # Check if last dim is 3 (for vectors)
        velocity_magnitude = np.linalg.norm(velocity_history_3d, axis=-1)
        print(f"  Velocity (3D, vector) shape: {velocity_history_3d.shape}")
    else:
        # Assume it's already a scalar magnitude field (e.g., (D,H,W))
        velocity_magnitude = velocity_history_3d
        print(f"  Velocity (3D, scalar) shape: {velocity_history_3d.shape}")

    print(f"  Velocity magnitude min: {velocity_magnitude.min()}, max: {velocity_magnitude.max()}")
    print(f"  Velocity magnitude unique values: {np.unique(velocity_magnitude)}")

    if velocity_magnitude.max() > velocity_magnitude.min():
        velocity_contribution = (velocity_magnitude - velocity_magnitude.min()) / \
                                (velocity_magnitude.max() - velocity_magnitude.min())
        print(f"  Velocity contribution normalized. Min: {velocity_contribution.min()}, Max: {velocity_contribution.max()}")
    else:
        velocity_contribution = np.zeros_like(velocity_magnitude)
        print("  Velocity magnitude data is uniform, contribution set to zeros.")


    # --- Turbulence kinetic energy contribution ---
    print(f"  Turbulence (3D) shape: {turbulence_history_3d.shape}")
    print(f"  Turbulence min: {turbulence_history_3d.min()}, max: {turbulence_history_3d.max()}")
    print(f"  Turbulence unique values: {np.unique(turbulence_history_3d)}")

    turbulence_contribution = turbulence_history_3d
    if turbulence_contribution.max() > turbulence_contribution.min():
        turbulence_contribution = (turbulence_contribution - turbulence_contribution.min()) / \
                                  (turbulence_contribution.max() - turbulence_contribution.min())
        print(f"  Turbulence contribution normalized. Min: {turbulence_contribution.min()}, Max: {turbulence_contribution.max()}")
    else:
        turbulence_contribution = np.zeros_like(turbulence_contribution)
        print("  Turbulence data is uniform, contribution set to zeros.")


    # --- Combine all normalized contributions ---
    fluid_scalar_field = (pressure_contribution + velocity_contribution + turbulence_contribution) / 3.0
    print(f"  Combined fluid_scalar_field min: {fluid_scalar_field.min()}, max: {fluid_scalar_field.max()}")
    print(f"  Combined fluid_scalar_field unique values: {np.unique(fluid_scalar_field)}")

    # Apply a Gaussian filter to smooth the scalar field.
    fluid_scalar_field_smoothed = gaussian_filter(fluid_scalar_field, sigma=0.5)
    print(f"  Smoothed fluid_scalar_field_smoothed min: {fluid_scalar_field_smoothed.min()}, max: {fluid_scalar_field_smoothed.max()}")
    print(f"  Smoothed fluid_scalar_field_smoothed unique values: {np.unique(fluid_scalar_field_smoothed)}")

    print("--- End Debugging extract_fluid_surface ---\n")

    return fluid_scalar_field_smoothed


def generate_mesh(surface_field, grid_metadata): # Changed nodes_coords to grid_metadata
    """
    Converts the extracted scalar field into a triangular mesh using the Marching Cubes algorithm.
    Vertices are then transformed into physical space using grid metadata.
    """

    # Debugging: Print shape and value range before attempting Marching Cubes
    print(f"🔍 Debugging: surface_field shape = {surface_field.shape}")
    print(f"🔍 Unique values in surface_field: {np.unique(surface_field)}")
    print(f"🔍 surface_field min = {surface_field.min()}, max = {surface_field.max()}")

    # Ensure the input is a valid 3D array for Marching Cubes.
    # If it's 4D (time series), select the first time step.
    if surface_field.ndim == 4:
        print(f"⚠️ Input is 4D. Selecting the first time step...")
        surface_field = surface_field[0]  # Extract single 3D frame for static mesh
    elif surface_field.ndim > 3:
        print(f"⚠️ Input is higher than 3D. Averaging over extra dimensions...")
        surface_field = np.mean(surface_field, axis=0) # Average along other dimensions
    elif surface_field.ndim < 3:
        raise ValueError(f"❌ Error: Expected 3D input but found {surface_field.ndim}D array for Marching Cubes.")

    print(f"✅ Fixed shape: surface_field = {surface_field.shape}")

    # Dynamically determine the isosurface level for Marching Cubes.
    # A common approach for a normalized "fluidness" field is the midpoint.
    min_val, max_val = surface_field.min(), surface_field.max()

    if min_val == max_val:
        raise RuntimeError("❌ No valid surface detected: All values in surface_field are identical. "
                           "This indicates uniform input data or an issue with fluid surface extraction logic.")

    # Set the surface level to the midpoint of the scalar field's range.
    # This is a good starting point for identifying the fluid-air interface.
    surface_level = (max_val + min_val) / 2.0

    # Ensure the chosen surface_level is strictly within the data's range to avoid issues.
    # Adding a small epsilon to prevent marching_cubes from failing on boundary conditions.
    epsilon = 1e-7 # A small constant epsilon
    # Check if range is effectively zero
    if (max_val - min_val) < 1e-6: # Using a small tolerance for "effectively zero range"
        # If range is too small, setting a level can be problematic.
        # This case should ideally be caught by `min_val == max_val` earlier,
        # but as a safeguard, make sure epsilon doesn't push it out of bounds.
        surface_level = min_val + (max_val - min_val) / 2.0 # Still midpoint
    else:
        surface_level = np.clip(surface_level, min_val + epsilon, max_val - epsilon)


    print(f"✅ Adjusted surface level: {surface_level}")

    # Extract surface mesh using the Marching Cubes algorithm.
    verts_grid_coords, faces, normals, _ = marching_cubes(surface_field, level=surface_level)

    # Check if Marching Cubes produced any geometry.
    if len(verts_grid_coords) == 0 or len(faces) == 0:
        raise RuntimeError(f"❌ Marching Cubes produced no geometry at level={surface_level}. "
                           "This may indicate the chosen surface_level is inappropriate, or the data is too uniform.")

    # Map vertices from grid coordinates to physical space using grid_metadata.
    try:
        # Retrieve grid dimensions, origin, and spacing from metadata.
        # Ensure your 'grid_metadata.json' contains these keys.
        grid_origin = np.array(grid_metadata['origin'])
        grid_spacing = np.array(grid_metadata['spacing'])

        # Transform grid coordinates (from Marching Cubes) to physical coordinates.
        verts_transformed = grid_origin + verts_grid_coords * grid_spacing
    except KeyError as e:
        raise RuntimeError(f"❌ Missing critical grid metadata for coordinate transformation: {e}. "
                           "Ensure 'origin' and 'spacing' are present in grid_metadata.json.")
    except Exception as e:
        # Catch any other potential errors during transformation.
        raise RuntimeError(f"❌ Error transforming vertices to physical space: {e}")

    return verts_transformed, faces, normals


def export_to_alembic(verts, faces, output_file):
    """
    Converts mesh data (vertices and faces) into the Alembic (.abc) format,
    which is suitable for animation and rendering pipelines.
    """
    # Create an Alembic archive at the specified output file path.
    archive = alembic.AbcGeom.OArchive(output_file)
    # Create a PolyMesh object within the Alembic archive.
    mesh_obj = alembic.AbcGeom.OPolyMesh(archive.getTop(), "fluid_mesh")

    # Get the mesh schema to store geometry data.
    mesh_schema = mesh_obj.getSchema()
    # Create a sample of the mesh geometry with vertices and faces.
    mesh_sample = alembic.AbcGeom.OPolyMeshSchemaSample(verts, faces)
    # Set the mesh sample to the schema, writing the data to the Alembic file.
    mesh_schema.set(mesh_sample)

    print(f"✅ Fluid surface mesh exported to {output_file}")


# Main execution block
if __name__ == "__main__":
    # Determine the workspace path, typically provided by GitHub Actions.
    workspace = os.getenv("GITHUB_WORKSPACE", ".")
    # Define the output path for the Alembic file.
    output_file = os.path.join(workspace, "data/testing-input-output/fluid_mesh.abc")

    print("🚀 Starting fluid mesh generation...")

    # Load all necessary simulation data.
    grid_metadata, nodes_coords, velocity_history, pressure_history, turbulence_history = load_simulation_data(workspace)

    # Extract the fluid surface by creating a scalar field representing fluidness.
    # This field will have continuous values, allowing Marching Cubes to find a surface.
    fluid_scalar_field = extract_fluid_surface(pressure_history, velocity_history, turbulence_history)

    # Generate the mesh (vertices, faces, normals) from the scalar field.
    # Pass grid_metadata to correctly transform vertices to physical coordinates.
    verts, faces, normals = generate_mesh(fluid_scalar_field, grid_metadata) # Pass grid_metadata here

    # Export the generated mesh to Alembic format.
    export_to_alembic(verts, faces, output_file)

    print("🎉 Fluid mesh generation completed successfully!")
