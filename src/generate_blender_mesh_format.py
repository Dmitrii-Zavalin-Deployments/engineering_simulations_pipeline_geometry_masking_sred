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

    # --- Normalize and combine contributions from different physical quantities ---

    # Pressure contribution: Assuming higher pressure indicates fluid regions.
    # Normalize to [0, 1] range.
    pressure_contribution = pressure_history
    if pressure_contribution.max() > pressure_contribution.min():
        pressure_contribution = (pressure_contribution - pressure_contribution.min()) / \
                                (pressure_contribution.max() - pressure_contribution.min())
    else:
        pressure_contribution = np.zeros_like(pressure_contribution) # Handle uniform data


    # Velocity magnitude contribution: Higher velocity might indicate active fluid regions.
    # Normalize to [0, 1] range.
    velocity_magnitude = np.linalg.norm(velocity_history, axis=-1)
    if velocity_magnitude.max() > velocity_magnitude.min():
        velocity_contribution = (velocity_magnitude - velocity_magnitude.min()) / \
                                (velocity_magnitude.max() - velocity_magnitude.min())
    else:
        velocity_contribution = np.zeros_like(velocity_magnitude)


    # Turbulence kinetic energy contribution: Can indicate turbulent fluid regions, adding detail.
    # Normalize to [0, 1] range.
    turbulence_contribution = turbulence_history
    if turbulence_contribution.max() > turbulence_contribution.min():
        turbulence_contribution = (turbulence_contribution - turbulence_contribution.min()) / \
                                  (turbulence_contribution.max() - turbulence_contribution.min())
    else:
        turbulence_contribution = np.zeros_like(turbulence_contribution)


    # Combine all normalized contributions to create a composite fluid scalar field.
    # A simple average is used here; weighted averages could be explored for fine-tuning.
    fluid_scalar_field = (pressure_contribution + velocity_contribution + turbulence_contribution) / 3.0

    # Apply a Gaussian filter to smooth the scalar field. This helps Marching Cubes
    # produce a cleaner, less noisy mesh. Sigma 0.5 is a light smoothing.
    fluid_scalar_field_smoothed = gaussian_filter(fluid_scalar_field, sigma=0.5)

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
    epsilon = 1e-5 * (max_val - min_val) # A small fraction of the total range
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
