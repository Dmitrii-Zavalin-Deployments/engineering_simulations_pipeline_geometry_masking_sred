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


def extract_fluid_surface(pressure_frame, velocity_frame, turbulence_frame):
    """
    Identifies the fluid surface by creating a binary mask based on
    the presence of any significant physical quantity, then smoothing it.
    This is robust for sparse or mostly zero simulation data, like a thin fluid flow.
    """
    # Define a small threshold for considering a value 'present'.
    # This is crucial for sparse data. Adjust if your minimum meaningful values are different.
    presence_threshold = 1e-6 # A very small value to catch any non-zero activity

    # Create binary masks where values exceed the threshold (using absolute value for robustness)
    pressure_mask = (np.abs(pressure_frame) > presence_threshold).astype(float)

    # Velocity magnitude: Check if it's a vector field (last dimension is 3) or scalar.
    if velocity_frame.ndim == 4 and velocity_frame.shape[-1] == 3:
        velocity_magnitude = np.linalg.norm(velocity_frame, axis=-1)
    else:
        velocity_magnitude = velocity_frame # Assume it's already a scalar magnitude
    velocity_mask = (np.abs(velocity_magnitude) > presence_threshold).astype(float)

    turbulence_mask = (np.abs(turbulence_frame) > presence_threshold).astype(float)

    # Combine masks: A voxel is considered fluid if *any* of the quantities are present.
    # We use 'maximum' to achieve an 'OR' operation for binary masks.
    fluid_scalar_field = np.maximum(pressure_mask, np.maximum(velocity_mask, turbulence_mask))

    # Apply a Gaussian filter to smooth the binary field into a continuous scalar field.
    # Sigma=1.0 is a good starting point for smoothing out blocky binary data into a renderable surface.
    fluid_scalar_field_smoothed = gaussian_filter(fluid_scalar_field, sigma=1.0)

    print(f"  Combined binary field (before smooth) min: {fluid_scalar_field.min()}, max: {fluid_scalar_field.max()}")
    print(f"  Smoothed scalar field shape: {fluid_scalar_field_smoothed.shape}, min: {fluid_scalar_field_smoothed.min()}, max: {fluid_scalar_field_smoothed.max()}")

    return fluid_scalar_field_smoothed


def generate_mesh(surface_field, grid_metadata):
    """
    Converts the extracted scalar field into a triangular mesh using the Marching Cubes algorithm.
    Vertices are then transformed into physical space using grid metadata.
    """
    # Ensure the input is a valid 3D array for Marching Cubes.
    if surface_field.ndim != 3:
        if surface_field.ndim == 4:
            surface_field = surface_field[0]
            print(f"⚠️ Warning: generate_mesh received 4D data, processed only first frame. Shape: {surface_field.shape}")
        else:
            raise ValueError(f"❌ Error: Expected 3D input but found {surface_field.ndim}D array in generate_mesh.")

    min_val, max_val = surface_field.min(), surface_field.max()
    print(f"  Marching Cubes input field min: {min_val}, max: {max_val}")

    # Check for uniform data before attempting Marching Cubes
    if min_val == max_val:
        raise RuntimeError("❌ No valid surface detected: All values in surface_field are identical. "
                           "This indicates uniform input data or an issue with fluid surface extraction logic for this frame.")

    # Set the isosurface level for Marching Cubes.
    # After smoothing the binary mask, the field will typically range from 0 to something near 1.
    # A level of 0.5 is a robust choice to capture the transition from 'empty' (values near 0) to 'fluid' (values near 1).
    surface_level = 0.5
    # Ensure the chosen surface_level is strictly within the data's range to prevent errors.
    epsilon = 1e-7 # A small constant epsilon
    surface_level_actual = np.clip(surface_level, min_val + epsilon, max_val - epsilon)

    print(f"  Marching Cubes level set to: {surface_level_actual}")

    # Extract surface mesh using Marching Cubes
    try:
        verts_grid_coords, faces, normals, _ = marching_cubes(surface_field, level=surface_level_actual)
        print(f"  Marching Cubes raw output - Vertices: {len(verts_grid_coords)}, Faces: {len(faces)}")
    except ValueError as e:
        # Marching cubes can raise ValueError for degenerate cases
        raise RuntimeError(f"❌ Marching Cubes failed for this frame at level={surface_level_actual}: {e}")

    # Check if Marching Cubes produced any geometry.
    if len(verts_grid_coords) == 0 or len(faces) == 0:
        raise RuntimeError(f"❌ Marching Cubes produced no geometry at level={surface_level_actual}. "
                           "This may indicate the chosen surface_level is inappropriate, or the data is too sparse for this frame.")

    # Map vertices from grid coordinates to physical space using grid_metadata.
    try:
        grid_origin = np.array(grid_metadata['origin'])
        grid_spacing = np.array(grid_metadata['spacing'])
        verts_transformed = grid_origin + verts_grid_coords * grid_spacing
    except KeyError as e:
        raise RuntimeError(f"❌ Missing critical grid metadata for coordinate transformation: {e}. "
                           "Ensure 'origin' and 'spacing' are present in grid_metadata.json.")
    except Exception as e:
        raise RuntimeError(f"❌ Error transforming vertices to physical space: {e}")

    return verts_transformed, faces, normals


def export_to_alembic(all_frame_data, output_file, total_frames_expected=1):
    """
    Converts a list of mesh data (vertices and faces for each frame) into an animated
    Alembic (.abc) file. Ensures an ABC file is always created, even if no valid mesh data
    was generated for any frame, by exporting a placeholder empty mesh.
    """
    print(f"Attempting to create Alembic archive at: {output_file}")
    try:
        archive = alembic.AbcGeom.OArchive(output_file)
        print("Alembic archive created successfully.")
    except Exception as e:
        print(f"❌ Error creating Alembic archive: {e}")
        print("🛑 Failed to create Alembic archive. Cannot proceed with export.")
        return # Cannot proceed if archive creation fails

    mesh_obj = alembic.AbcGeom.OPolyMesh(archive.getTop(), "fluid_mesh")
    mesh_schema = mesh_obj.getSchema()

    fps = 24.0 # Frames per second for the animation
    time_sampling = alembic.Abc.TimeSampling(1.0 / fps, alembic.Abc.TimeSamplingType.kUniformType)
    mesh_schema.setTimeSampling(time_sampling)

    if not all_frame_data:
        print(f"⚠️ No valid mesh data generated for any frame. Exporting an empty/placeholder Alembic file with {total_frames_expected} frames.")
        # If no valid data, export a single empty mesh sample for each expected frame.
        empty_verts = np.array([], dtype=np.float32).reshape(0, 3) # Empty array of shape (0,3)
        empty_faces = np.array([], dtype=np.int32).reshape(0, 3)   # Empty array of shape (0,3)

        # Create an empty sample for each expected frame to maintain animation length in Blender
        for i in range(total_frames_expected):
            frame_time = i / fps
            mesh_sample = alembic.AbcGeom.OPolyMeshSchemaSample(empty_verts, empty_faces)
            mesh_schema.set(mesh_sample, frame_time)

        print(f"✅ Empty Alembic fluid surface mesh exported to {output_file}.")
    else:
        # Sort all_frame_data by time to ensure samples are added in chronological order.
        all_frame_data.sort(key=lambda x: x[0])
        for frame_time_in_seconds, verts, faces in all_frame_data:
            mesh_sample = alembic.AbcGeom.OPolyMeshSchemaSample(verts, faces)
            mesh_schema.set(mesh_sample, frame_time_in_seconds) # Set sample at specific time
        print(f"✅ Animated fluid surface mesh exported to {output_file} with {len(all_frame_data)} actual mesh frames.")

    # The archive is automatically closed when it goes out of scope (e.g., function returns).


# Main execution block
if __name__ == "__main__":
    workspace = os.getenv("GITHUB_WORKSPACE", ".")
    output_file = os.path.join(workspace, "data/testing-input-output/fluid_mesh.abc")

    print("🚀 Starting fluid mesh generation and animation export...")

    grid_metadata, nodes_coords, velocity_history, pressure_history, turbulence_history = load_simulation_data(workspace)

    # Determine the number of time steps from the loaded history data.
    num_time_steps = pressure_history.shape[0] # Assuming all histories have same time dimension

    # List to store successfully generated mesh data for each frame.
    all_mesh_frames_data = []

    print(f"Processing {num_time_steps} time steps...")

    for i in range(num_time_steps):
        print(f"\n--- Processing frame {i} ---")
        # Extract 3D data for the current time step.
        pressure_frame = pressure_history[i]
        velocity_frame = velocity_history[i]
        turbulence_frame = turbulence_history[i]

        verts, faces, normals = None, None, None # Initialize to None for clear error handling.

        try:
            # Attempt to extract fluid surface from the current frame's data.
            fluid_scalar_field = extract_fluid_surface(pressure_frame, velocity_frame, turbulence_frame)
            # Attempt to generate a mesh from the extracted scalar field.
            verts, faces, normals = generate_mesh(fluid_scalar_field, grid_metadata)

            # Store the data for this frame ONLY if valid mesh geometry was produced.
            if verts is not None and faces is not None and len(verts) > 0 and len(faces) > 0:
                # Assuming frame 'i' corresponds to time 'i / fps' seconds.
                # Adjust '24.0' if your simulation's effective frame rate is different.
                frame_time = i / 24.0
                all_mesh_frames_data.append((frame_time, verts, faces))
                print(f"✅ Mesh generated successfully for frame {i}. Verts: {len(verts)}, Faces: {len(faces)}")
            else:
                # This catches cases where marching_cubes runs without error but produces no geometry.
                print(f"⚠️ Marching Cubes generated an empty mesh for frame {i} despite no RuntimeError. Skipping this frame.")

        except RuntimeError as e:
            # Catch specific errors from extract_fluid_surface or generate_mesh.
            print(f"⚠️ Skipping frame {i} due to a processing error: {e}")
            # No mesh data is added for this frame. The export_to_alembic will handle gaps.

        except Exception as e:
            # Catch any other unexpected errors during frame processing.
            print(f"❌ An unexpected error occurred while processing frame {i}: {e}")

    print(f"\n--- Mesh Generation Summary ---")
    print(f"Total number of frames successfully processed and collected: {len(all_mesh_frames_data)}")

    # Export all collected mesh data to an animated Alembic file.
    # Pass the total number of expected frames to ensure the ABC file spans the full animation length.
    export_to_alembic(all_mesh_frames_data, output_file, total_frames_expected=num_time_steps)

    print("🎉 Fluid mesh generation and animation export process completed!")
