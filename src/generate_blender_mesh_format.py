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
    Identifies the fluid surface by creating a scalar field representing "fluidness."
    Higher values in the field indicate a higher likelihood of being fluid.
    This field is then smoothed for better mesh generation.
    It now expects a single 3D frame, not history.
    """
    # Now this function operates on a single 3D frame, so no need for [0] indexing here.

    # --- Debugging (can be removed once stable) ---
    # print("\n--- Debugging extract_fluid_surface (per frame) ---")
    # print(f"  Pressure frame shape: {pressure_frame.shape}, min: {pressure_frame.min()}, max: {pressure_frame.max()}, unique: {np.unique(pressure_frame)}")
    # print(f"  Velocity frame shape: {velocity_frame.shape}, min: {velocity_frame.min()}, max: {velocity_frame.max()}")
    # if velocity_frame.shape[-1] == 3:
    #    print(f"  Velocity frame unique values (slice for brevity): {np.unique(velocity_frame[..., 0])}")
    # else:
    #    print(f"  Velocity frame unique values: {np.unique(velocity_frame)}")
    # print(f"  Turbulence frame shape: {turbulence_frame.shape}, min: {turbulence_frame.min()}, max: {turbulence_frame.max()}, unique: {np.unique(turbulence_frame)}")


    # --- Normalize and combine contributions from different physical quantities ---

    # Pressure contribution: Assuming higher pressure indicates fluid regions.
    pressure_contribution = pressure_frame
    if pressure_contribution.max() > pressure_contribution.min():
        pressure_contribution = (pressure_contribution - pressure_contribution.min()) / \
                                (pressure_contribution.max() - pressure_contribution.min())
    else:
        pressure_contribution = np.zeros_like(pressure_contribution) # Handle uniform data


    # Velocity magnitude contribution: Higher velocity might indicate active fluid regions.
    # Check if velocity_frame is a vector field (last dimension is 3) or scalar.
    if velocity_frame.ndim == 4 and velocity_frame.shape[-1] == 3:
        velocity_magnitude = np.linalg.norm(velocity_frame, axis=-1)
    else:
        velocity_magnitude = velocity_frame # Assume it's already a scalar magnitude

    if velocity_magnitude.max() > velocity_magnitude.min():
        velocity_contribution = (velocity_magnitude - velocity_magnitude.min()) / \
                                (velocity_magnitude.max() - velocity_magnitude.min())
    else:
        velocity_contribution = np.zeros_like(velocity_magnitude)


    # Turbulence kinetic energy contribution: Can indicate turbulent fluid regions, adding detail.
    turbulence_contribution = turbulence_frame
    if turbulence_contribution.max() > turbulence_contribution.min():
        turbulence_contribution = (turbulence_contribution - turbulence_contribution.min()) / \
                                  (turbulence_contribution.max() - turbulence_contribution.min())
    else:
        turbulence_contribution = np.zeros_like(turbulence_contribution)


    # Combine all normalized contributions to create a composite fluid scalar field.
    fluid_scalar_field = (pressure_contribution + velocity_contribution + turbulence_contribution) / 3.0

    # Apply a Gaussian filter to smooth the scalar field.
    fluid_scalar_field_smoothed = gaussian_filter(fluid_scalar_field, sigma=0.5)

    print(f"  Smoothed scalar field shape: {fluid_scalar_field_smoothed.shape}, min: {fluid_scalar_field_smoothed.min()}, max: {fluid_scalar_field_smoothed.max()}")
    # print("--- End Debugging extract_fluid_surface ---\n")

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

    # Set the isosurface level to the midpoint of the scalar field's range.
    surface_level = (max_val + min_val) / 2.0

    # Ensure the chosen surface_level is strictly within the data's range.
    epsilon = 1e-7 # A small constant epsilon
    if (max_val - min_val) < 1e-6: # Using a small tolerance for "effectively zero range"
        surface_level = min_val + (max_val - min_val) / 2.0
    else:
        surface_level = np.clip(surface_level, min_val + epsilon, max_val - epsilon)

    print(f"  Marching Cubes level set to: {surface_level}")

    # Extract surface mesh using Marching Cubes
    try:
        verts_grid_coords, faces, normals, _ = marching_cubes(surface_field, level=surface_level)
        print(f"  Marching Cubes raw output - Vertices: {len(verts_grid_coords)}, Faces: {len(faces)}")
    except ValueError as e:
        # Marching cubes can raise ValueError for degenerate cases
        raise RuntimeError(f"❌ Marching Cubes failed for this frame at level={surface_level}: {e}")

    # Check if Marching Cubes produced any geometry.
    if len(verts_grid_coords) == 0 or len(faces) == 0:
        raise RuntimeError(f"❌ Marching Cubes produced no geometry at level={surface_level}. "
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


def export_to_alembic(all_frame_data, output_file):
    """
    Converts a list of mesh data (vertices and faces for each frame) into an animated
    Alembic (.abc) file.
    """
    if not all_frame_data:
        print("⚠️ No valid mesh data generated to export to Alembic.")
        return

    print(f"Attempting to create Alembic archive at: {output_file}")
    try:
        archive = alembic.AbcGeom.OArchive(output_file)
        print("Alembic archive created successfully.")
    except Exception as e:
        print(f"❌ Error creating Alembic archive: {e}")
        return

    mesh_obj = alembic.AbcGeom.OPolyMesh(archive.getTop(), "fluid_mesh")
    mesh_schema = mesh_obj.getSchema()

    # Define time sampling: Assuming uniform time steps, e.g., 24 frames per second.
    # You might want to get dt from grid_metadata if available.
    fps = 24.0 # Frames per second for the animation
    time_sampling = alembic.Abc.TimeSampling(1.0 / fps, alembic.Abc.TimeSamplingType.kUniformType)
    mesh_schema.setTimeSampling(time_sampling)

    # Add each frame's mesh data as a sample to the Alembic file
    # Ensure that `all_frame_data` is a list of tuples: (time_in_seconds, verts, faces)
    for frame_time_in_seconds, verts, faces in all_frame_data:
        mesh_sample = alembic.AbcGeom.OPolyMeshSchemaSample(verts, faces)
        mesh_schema.set(mesh_sample, frame_time_in_seconds) # Set sample at specific time

    print(f"✅ Animated fluid surface mesh exported to {output_file} with {len(all_frame_data)} frames.")


# Main execution block
if __name__ == "__main__":
    workspace = os.getenv("GITHUB_WORKSPACE", ".")
    output_file = os.path.join(workspace, "data/testing-input-output/fluid_mesh.abc")

    print("🚀 Starting fluid mesh generation and animation export...")

    grid_metadata, nodes_coords, velocity_history, pressure_history, turbulence_history = load_simulation_data(workspace)

    # Determine the number of time steps
    num_time_steps = pressure_history.shape[0] # Assuming all histories have same time dim

    # List to store successfully generated mesh data for each frame
    all_mesh_frames_data = []

    print(f"Processing {num_time_steps} time steps...")

    for i in range(num_time_steps):
        print(f"\n--- Processing frame {i} ---")
        # Extract 3D data for the current time step
        pressure_frame = pressure_history[i]
        velocity_frame = velocity_history[i]
        turbulence_frame = turbulence_history[i]

        verts, faces, normals = None, None, None # Initialize to None for error handling clarity

        try:
            fluid_scalar_field = extract_fluid_surface(pressure_frame, velocity_frame, turbulence_frame)
            verts, faces, normals = generate_mesh(fluid_scalar_field, grid_metadata)

            # Store the data for this frame ONLY if valid mesh was generated.
            if verts is not None and faces is not None and len(verts) > 0 and len(faces) > 0:
                # Assuming frame 'i' corresponds to time 'i / fps' seconds.
                # You might need to adjust the time calculation based on your simulation's dt.
                frame_time = i / 24.0 # Example: if simulation runs at 24 frames per second
                all_mesh_frames_data.append((frame_time, verts, faces))
                print(f"✅ Mesh generated successfully for frame {i}. Verts: {len(verts)}, Faces: {len(faces)}")
            else:
                print(f"⚠️ Marching Cubes generated an empty mesh for frame {i} despite no RuntimeError. Skipping this frame.")


        except RuntimeError as e:
            print(f"⚠️ Skipping frame {i} due to a processing error: {e}")
            # Optionally, you could append the previous frame's data here to hold the last valid mesh
            # for a smoother animation if a frame is skipped.
            # if all_mesh_frames_data:
            #    all_mesh_frames_data.append(all_mesh_frames_data[-1]) # Append last valid frame
            # else:
            #    pass # No previous frame to append, just skip

        except Exception as e:
            print(f"❌ An unexpected error occurred while processing frame {i}: {e}")

    print(f"\n--- Mesh Generation Summary ---")
    print(f"Total number of frames successfully processed and collected: {len(all_mesh_frames_data)}")

    # Export all collected mesh data to an animated Alembic file
    export_to_alembic(all_mesh_frames_data, output_file)

    print("🎉 Fluid mesh generation and animation export process completed!")
