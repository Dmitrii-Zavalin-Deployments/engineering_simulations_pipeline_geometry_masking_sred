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
    # Construct the data directory path relative to the provided workspace
    data_dir = os.path.join(workspace, "data", "testing-input-output")

    # Verify that the data directory exists
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}. Please ensure simulation output files are in this location.")

    # Load metadata & simulation files with robust file existence checks
    grid_metadata_path = os.path.join(data_dir, "grid_metadata.json")
    if not os.path.exists(grid_metadata_path):
        raise FileNotFoundError(f"grid_metadata.json not found: {grid_metadata_path}")
    with open(grid_metadata_path, "r") as f:
        grid_metadata = json.load(f)

    nodes_coords_path = os.path.join(data_dir, "nodes_coords.npy")
    if not os.path.exists(nodes_coords_path):
        raise FileNotFoundError(f"nodes_coords.npy not found: {nodes_coords_path}")
    nodes_coords = np.load(nodes_coords_path)

    velocity_history_path = os.path.join(data_dir, "velocity_history.npy")
    if not os.path.exists(velocity_history_path):
        raise FileNotFoundError(f"velocity_history.npy not found: {velocity_history_path}")
    velocity_history = np.load(velocity_history_path)

    pressure_history_path = os.path.join(data_dir, "pressure_history.npy")
    if not os.path.exists(pressure_history_path):
        raise FileNotFoundError(f"pressure_history.npy not found: {pressure_history_path}")
    pressure_history = np.load(pressure_history_path)

    turbulence_history_path = os.path.join(data_dir, "turbulence_kinetic_energy_history.npy")
    if not os.path.exists(turbulence_history_path):
        raise FileNotFoundError(f"turbulence_kinetic_energy_history.npy not found: {turbulence_history_path}")
    turbulence_history = np.load(turbulence_history_path)

    return grid_metadata, nodes_coords, velocity_history, pressure_history, turbulence_history


def extract_fluid_surface(pressure_frame, velocity_frame, turbulence_frame):
    """
    Identifies the fluid surface by creating a binary mask based on
    the presence of any significant physical quantity, then smoothing it.
    This is robust for sparse or mostly zero simulation data, like a thin fluid flow.
    """
    # Define a small threshold for considering a value 'present'.
    presence_threshold = 1e-6

    # Create binary masks where values exceed the threshold (using absolute value for robustness)
    pressure_mask = (np.abs(pressure_frame) > presence_threshold).astype(float)

    # Velocity magnitude: Check if it's a vector field (last dimension is 3) or scalar.
    if velocity_frame.ndim == 4 and velocity_frame.shape[-1] == 3:
        velocity_magnitude = np.linalg.norm(velocity_frame, axis=-1)
    else:
        velocity_magnitude = velocity_frame
    velocity_mask = (np.abs(velocity_magnitude) > presence_threshold).astype(float)

    turbulence_mask = (np.abs(turbulence_frame) > presence_threshold).astype(float)

    # Combine masks: A voxel is considered fluid if *any* of the quantities are present.
    # We use 'maximum' to achieve an 'OR' operation for binary masks.
    fluid_scalar_field = np.maximum(pressure_mask, np.maximum(velocity_mask, turbulence_mask))

    # Apply a Gaussian filter to smooth the binary field into a continuous scalar field.
    fluid_scalar_field_smoothed = gaussian_filter(fluid_scalar_field, sigma=1.0)

    print(f"  Combined binary field (before smooth) min: {fluid_scalar_field.min()}, max: {fluid_scalar_field.max()}")
    print(f"  Smoothed scalar field shape: {fluid_scalar_field_smoothed.shape}, min: {fluid_scalar_field_smoothed.min()}, max: {fluid_scalar_field_smoothed.max()}")

    return fluid_scalar_field_smoothed


def generate_mesh(surface_field, grid_metadata):
    """
    Converts the extracted scalar field into a triangular mesh using the Marching Cubes algorithm.
    Vertices are then transformed into physical space using grid metadata.
    Returns (verts, faces, normals) or (empty_verts, empty_faces, empty_normals) if no mesh is found.
    """
    # Define empty arrays to return if no geometry is generated
    empty_verts = np.array([], dtype=np.float32).reshape(0, 3)
    empty_faces = np.array([], dtype=np.int32).reshape(0, 3)
    empty_normals = np.array([], dtype=np.float32).reshape(0, 3)

    # Ensure the input is a valid 3D array for Marching Cubes.
    if surface_field.ndim != 3:
        if surface_field.ndim == 4:
            surface_field = surface_field[0]
            print(f"⚠️ Warning: generate_mesh received 4D data, processed only first frame. Shape: {surface_field.shape}")
        else:
            print(f"❌ Error: Expected 3D input but found {surface_field.ndim}D array in generate_mesh. Returning empty mesh.")
            return empty_verts, empty_faces, empty_normals

    min_val, max_val = surface_field.min(), surface_field.max()
    print(f"  Marching Cubes input field min: {min_val}, max: {max_val}")

    # If the field is uniform, Marching Cubes cannot extract a surface.
    if min_val == max_val:
        print("❌ No valid surface detected: All values in surface_field are identical. Returning empty mesh.")
        return empty_verts, empty_faces, empty_normals

    # Set the isosurface level for Marching Cubes.
    surface_level = 0.5
    # Ensure the chosen surface_level is strictly within the data's range to prevent errors.
    epsilon = 1e-7
    surface_level_actual = np.clip(surface_level, min_val + epsilon, max_val - epsilon)

    print(f"  Marching Cubes level set to: {surface_level_actual}")

    # Extract surface mesh using Marching Cubes
    try:
        verts_grid_coords, faces, normals, _ = marching_cubes(surface_field, level=surface_level_actual)
        print(f"  Marching Cubes raw output - Vertices: {len(verts_grid_coords)}, Faces: {len(faces)}")
    except ValueError as e:
        # Marching cubes can raise ValueError for degenerate cases. Return empty mesh gracefully.
        print(f"❌ Marching Cubes failed for this frame at level={surface_level_actual}: {e}. Returning empty mesh.")
        return empty_verts, empty_faces, empty_normals

    # Check if Marching Cubes produced any geometry. If not, return empty.
    if len(verts_grid_coords) == 0 or len(faces) == 0:
        print(f"❌ Marching Cubes produced no geometry at level={surface_level_actual}. Returning empty mesh.")
        return empty_verts, empty_faces, empty_normals

    # Map vertices from grid coordinates to physical space using grid_metadata.
    try:
        grid_origin = np.array(grid_metadata['origin'])
        grid_spacing = np.array(grid_metadata['spacing'])
        verts_transformed = grid_origin + verts_grid_coords * grid_spacing
    except KeyError as e:
        print(f"❌ Missing critical grid metadata for coordinate transformation: {e}. Returning empty mesh.")
        return empty_verts, empty_faces, empty_normals
    except Exception as e:
        print(f"❌ Error transforming vertices to physical space: {e}. Returning empty mesh.")
        return empty_verts, empty_faces, empty_normals

    return verts_transformed, faces, normals


def export_to_alembic(all_frame_data, output_file, total_frames_expected=1):
    """
    Converts a list of mesh data (vertices and faces for each frame) into an animated
    Alembic (.abc) file. Ensures an ABC file is always created, even if no valid mesh data
    was generated for any frame, by exporting a placeholder empty mesh.
    """
    print(f"Attempting to create Alembic archive at: {output_file}")

    # Ensure the output directory exists before attempting to create the archive
    output_dir = os.path.dirname(output_file)
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Ensured output directory exists: {output_dir}")
    except OSError as e:
        print(f"❌ Error creating output directory {output_dir}: {e}")
        print("🛑 Cannot proceed with Alembic export as directory creation failed.")
        return # Exit if directory creation fails

    try:
        archive = alembic.AbcGeom.OArchive(output_file)
        print("Alembic archive created successfully.")
    except Exception as e:
        print(f"❌ Error creating Alembic archive at {output_file}: {e}")
        print("🛑 Failed to create Alembic archive. This might indicate a deeper issue with the Alembic library or file system.")
        return # Cannot proceed if archive creation fails

    mesh_obj = alembic.AbcGeom.OPolyMesh(archive.getTop(), "fluid_mesh")
    mesh_schema = mesh_obj.getSchema()

    fps = 24.0 # Frames per second for the animation
    time_sampling = alembic.Abc.TimeSampling(1.0 / fps, alembic.Abc.TimeSamplingType.kUniformType)
    mesh_schema.setTimeSampling(time_sampling)

    if not all_frame_data:
        print(f"⚠️ No valid mesh data generated for any frame. Exporting an empty/placeholder Alembic file with {total_frames_expected} frames.")
        empty_verts = np.array([], dtype=np.float32).reshape(0, 3)
        empty_faces = np.array([], dtype=np.int32).reshape(0, 3)

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
            # Ensure verts and faces are valid NumPy arrays for Alembic
            if not isinstance(verts, np.ndarray) or not isinstance(faces, np.ndarray):
                 print(f"⚠️ Invalid data type for verts or faces at time {frame_time_in_seconds}. Skipping sample.")
                 continue # Skip this sample if data types are incorrect

            mesh_sample = alembic.AbcGeom.OPolyMeshSchemaSample(verts, faces)
            mesh_schema.set(mesh_sample, frame_time_in_seconds) # Set sample at specific time
        print(f"✅ Animated fluid surface mesh exported to {output_file} with {len(all_frame_data)} actual mesh frames.")

    # The archive is automatically closed when it goes out of scope (e.g., function returns).


# Main execution block
if __name__ == "__main__":
    # Get the base path of the repository. In GitHub Actions, this is GITHUB_WORKSPACE.
    # When running the script from src/, os.getcwd() will be the src/ directory.
    # The workflow's 'Run Fluid Mesh Processing Script' step changes directory to src/
    base_dir_from_script = os.getcwd() # This will be /home/runner/work/.../engineering_simulations_pipeline_mesh_sred/src

    # To get to the repository root, we need to go up one level.
    # The output file is at data/testing-input-output relative to the repository root.
    repo_root = os.path.dirname(base_dir_from_script) # This will be /home/runner/work/.../engineering_simulations_pipeline_mesh_sred
    
    output_file = os.path.join(repo_root, "data", "testing-input-output", "fluid_mesh.abc")

    print("🚀 Starting fluid mesh generation and animation export...")
    print(f"  Script's current working directory: {base_dir_from_script}")
    print(f"  Calculated repository root: {repo_root}")
    print(f"  Expected output file path: {output_file}")

    # Load simulation data, ensuring it uses the correct repository root for input files
    try:
        grid_metadata, nodes_coords, velocity_history, pressure_history, turbulence_history = load_simulation_data(repo_root)
    except FileNotFoundError as e:
        print(f"🛑 Error loading simulation data: {e}")
        print("Exiting as essential input files are missing.")
        exit(1) # Exit the script with an error code

    # Determine the number of time steps from the loaded history data.
    num_time_steps = pressure_history.shape[0] # Assuming all histories have same time dimension

    # List to store successfully generated mesh data for each frame.
    all_mesh_frames_data = []

    print(f"Processing {num_time_steps} time steps...")
    
    frames_with_geometry = 0

    for i in range(num_time_steps):
        print(f"\n--- Processing frame {i} ---")
        # Extract 3D data for the current time step.
        pressure_frame = pressure_history[i]
        velocity_frame = velocity_history[i]
        turbulence_frame = turbulence_history[i]

        try:
            fluid_scalar_field = extract_fluid_surface(pressure_frame, velocity_frame, turbulence_frame)
            verts, faces, normals = generate_mesh(fluid_scalar_field, grid_metadata)

            # Always add a frame, whether it has geometry or is empty.
            frame_time = i / 24.0 # Assuming 24 FPS
            all_mesh_frames_data.append((frame_time, verts, faces))
            
            if len(verts) > 0 and len(faces) > 0:
                print(f"✅ Mesh generated successfully for frame {i}. Verts: {len(verts)}, Faces: {len(faces)}")
                frames_with_geometry += 1
            else:
                print(f"✅ Mesh generated (empty/placeholder) for frame {i}.")

        except Exception as e:
            # If any unhandled error occurs during frame processing, still add an empty frame
            print(f"❌ An error occurred while processing frame {i}: {e}")
            empty_verts = np.array([], dtype=np.float32).reshape(0, 3)
            empty_faces = np.array([], dtype=np.int32).reshape(0, 3)
            frame_time = i / 24.0
            all_mesh_frames_data.append((frame_time, empty_verts, empty_faces))
            print(f"⚠️ Added empty mesh for frame {i} due to error to maintain animation length.")

    print(f"\n--- Mesh Generation Summary ---")
    print(f"Total number of frames collected (including empty): {len(all_mesh_frames_data)}")
    print(f"Number of frames with actual geometry: {frames_with_geometry}")

    # Export all collected mesh data to an animated Alembic file.
    export_to_alembic(all_mesh_frames_data, output_file, total_frames_expected=num_time_steps)

    print("🎉 Fluid mesh generation and animation export process completed!")
