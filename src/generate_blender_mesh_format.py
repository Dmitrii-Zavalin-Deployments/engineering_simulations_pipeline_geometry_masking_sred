import json
import numpy as np
import os

def generate_fluid_mesh_data_json(
    navier_stokes_results_path,
    output_mesh_json_path="data/testing-input-output/fluid_mesh_data.json"
):
    """
    Processes fluid simulation data to extract outer boundary nodes as a mesh
    and saves their animated positions and static faces into a JSON file.

    Args:
        navier_stokes_results_path (str): Path to the navier_stokes_results.json file.
        output_mesh_json_path (str): Path to save the generated fluid_mesh_data.json file.
    """
    print(f"Loading data from {navier_stokes_results_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_mesh_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 1. Load Input Data
    try:
        with open(navier_stokes_results_path, 'r') as f:
            navier_stokes_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}. Please ensure path is correct.")
        # For testing, it's better to raise an exception here to signal failure
        raise FileNotFoundError(f"Input file not found: {e}")
    # REMOVED the specific 'except json.JSONDecodeError' block.
    # json.JSONDecodeError will now propagate directly from json.load(f) if the file is malformed.


    # --- Input Validation Start ---

    # Validate 'time_points' existence and type
    if 'time_points' not in navier_stokes_data:
        raise KeyError("Input JSON is missing 'time_points' key.")
    if not isinstance(navier_stokes_data['time_points'], list):
        raise TypeError("'time_points' must be a list.")
    time_points = np.array(navier_stokes_data['time_points'])

    # Validate 'time_points' for monotonicity
    if len(time_points) > 1: # Only check for monotonicity if there's more than one time point
        for i in range(1, len(time_points)):
            if time_points[i] <= time_points[i-1]:
                raise ValueError(
                    f"Time points are not monotonically increasing. Issue at index {i}: "
                    f"{time_points[i-1]} followed by {time_points[i]}."
                )

    # Validate 'mesh_info' existence and its sub-keys
    if 'mesh_info' not in navier_stokes_data:
        raise KeyError("Input JSON is missing 'mesh_info' key.")
    
    mesh_info = navier_stokes_data['mesh_info'] # Access mesh_info once

    if 'nodes_coords' not in mesh_info:
        raise KeyError("Input JSON is missing 'mesh_info.nodes_coords' key.")
    if 'grid_shape' not in mesh_info:
        raise KeyError("Input JSON is missing 'mesh_info.grid_shape' key.")

    initial_nodes_coords = np.array(mesh_info['nodes_coords'])
    grid_shape = mesh_info['grid_shape'] # [Z, Y, X]

    # Validate grid_shape: must be 3 integers, and positive
    if not (isinstance(grid_shape, list) and len(grid_shape) == 3 and
            all(isinstance(dim, int) and dim > 0 for dim in grid_shape)):
        raise ValueError(
            f"Invalid 'grid_shape' in mesh_info. Expected a list of 3 positive integers, got {grid_shape}."
        )

    num_z, num_y, num_x = grid_shape

    # Validate nodes_coords size against grid_shape
    expected_total_nodes = num_z * num_y * num_x
    if len(initial_nodes_coords) != expected_total_nodes:
        raise ValueError(
            f"Mismatch between 'nodes_coords' size ({len(initial_nodes_coords)}) "
            f"and 'grid_shape' calculated total nodes ({expected_total_nodes})."
        )
    
    # Validate 'velocity_history' existence
    if 'velocity_history' not in navier_stokes_data:
        raise KeyError("Input JSON is missing 'velocity_history' key.")
    velocity_history = navier_stokes_data['velocity_history'] # Per time step, per node

    # Validate velocity_history structure and size
    if not isinstance(velocity_history, list) or len(velocity_history) != len(time_points):
        raise ValueError(
            f"Velocity history must be a list with the same number of entries as time points ({len(time_points)})."
        )
    for t_idx, velocities_at_t in enumerate(velocity_history):
        if not isinstance(velocities_at_t, list) or len(velocities_at_t) != expected_total_nodes:
            raise ValueError(
                f"Velocity data for time step {t_idx} has incorrect number of entries. "
                f"Expected {expected_total_nodes}, got {len(velocities_at_t)}."
            )
        # Optional: Further check if each velocity entry is a 3-element list of numbers
        for node_vel in velocities_at_t:
            if not (isinstance(node_vel, list) and len(node_vel) == 3 and
                    all(isinstance(v, (int, float)) for v in node_vel)):
                raise ValueError(
                    f"Velocity entry for a node is malformed at time step {t_idx}. Expected [vx, vy, vz] of numbers."
                )

    # --- Input Validation End ---
    
    # --- Identify Surface Nodes and Create Static Faces ---
    def get_1d_index(ix, iy, iz, nx, ny):
        return iz * (ny * nx) + iy * nx + ix

    boundary_1d_indices = set()
    for iz in range(num_z):
        for iy in range(num_y):
            for ix in range(num_x):
                is_boundary = (ix == 0 or ix == num_x - 1 or
                               iy == 0 or iy == num_y - 1 or
                               iz == 0 or iz == num_z - 1)
                if is_boundary:
                    idx_1d = get_1d_index(ix, iy, iz, num_x, num_y)
                    boundary_1d_indices.add(idx_1d)

    sorted_boundary_indices = sorted(list(boundary_1d_indices))
    global_to_local_idx_map = {global_idx: local_idx for local_idx, global_idx in enumerate(sorted_boundary_indices)}
    
    static_faces = []
    
    def add_quad_face(face_list, p0, p1, p2, p3):
        face_list.append([p0, p1, p2, p3])

    # Iterate over ZY faces (fixed X)
    for iz in range(num_z - 1):
        for iy in range(num_y - 1):
            if 0 in [ix for ix in range(num_x)]: # Min X face
                p0_global = get_1d_index(0, iy, iz, num_x, num_y)
                p1_global = get_1d_index(0, iy + 1, iz, num_x, num_y)
                p2_global = get_1d_index(0, iy + 1, iz + 1, num_x, num_y)
                p3_global = get_1d_index(0, iy, iz + 1, num_x, num_y)
                if all(p in boundary_1d_indices for p in [p0_global, p1_global, p2_global, p3_global]):
                    add_quad_face(static_faces,
                                  global_to_local_idx_map[p0_global],
                                  global_to_local_idx_map[p1_global],
                                  global_to_local_idx_map[p2_global],
                                  global_to_local_idx_map[p3_global])
            if (num_x - 1) in [ix for ix in range(num_x)]: # Max X face
                p0_global = get_1d_index(num_x - 1, iy, iz, num_x, num_y)
                p1_global = get_1d_index(num_x - 1, iy + 1, iz, num_x, num_y)
                p2_global = get_1d_index(num_x - 1, iy + 1, iz + 1, num_x, num_y)
                p3_global = get_1d_index(num_x - 1, iy, iz + 1, num_x, num_y)
                if all(p in boundary_1d_indices for p in [p0_global, p1_global, p2_global, p3_global]):
                    add_quad_face(static_faces,
                                  global_to_local_idx_map[p0_global],
                                  global_to_local_idx_map[p1_global],
                                  global_to_local_idx_map[p2_global],
                                  global_to_local_idx_map[p3_global])

    # Iterate over ZX faces (fixed Y)
    for iz in range(num_z - 1):
        for ix in range(num_x - 1):
            if 0 in [iy for iy in range(num_y)]: # Min Y face
                p0_global = get_1d_index(ix, 0, iz, num_x, num_y)
                p1_global = get_1d_index(ix + 1, 0, iz, num_x, num_y)
                p2_global = get_1d_index(ix + 1, 0, iz + 1, num_x, num_y)
                p3_global = get_1d_index(ix, 0, iz + 1, num_x, num_y)
                if all(p in boundary_1d_indices for p in [p0_global, p1_global, p2_global, p3_global]):
                    add_quad_face(static_faces,
                                  global_to_local_idx_map[p0_global],
                                  global_to_local_idx_map[p1_global],
                                  global_to_local_idx_map[p2_global],
                                  global_to_local_idx_map[p3_global])
            if (num_y - 1) in [iy for iy in range(num_y)]: # Max Y face
                p0_global = get_1d_index(ix, num_y - 1, iz, num_x, num_y)
                p1_global = get_1d_index(ix + 1, num_y - 1, iz, num_x, num_y)
                p2_global = get_1d_index(ix + 1, num_y - 1, iz + 1, num_x, num_y)
                p3_global = get_1d_index(ix, num_y - 1, iz + 1, num_x, num_y)
                if all(p in boundary_1d_indices for p in [p0_global, p1_global, p2_global, p3_global]):
                    add_quad_face(static_faces,
                                  global_to_local_idx_map[p0_global],
                                  global_to_local_idx_map[p1_global],
                                  global_to_local_idx_map[p2_global],
                                  global_to_local_idx_map[p3_global])

    # Iterate over XY faces (fixed Z)
    for iy in range(num_y - 1):
        for ix in range(num_x - 1):
            if 0 in [iz for iz in range(num_z)]: # Min Z face
                p0_global = get_1d_index(ix, iy, 0, num_x, num_y)
                p1_global = get_1d_index(ix + 1, iy, 0, num_x, num_y)
                p2_global = get_1d_index(ix + 1, iy + 1, 0, num_x, num_y)
                p3_global = get_1d_index(ix, iy + 1, 0, num_x, num_y)
                if all(p in boundary_1d_indices for p in [p0_global, p1_global, p2_global, p3_global]):
                    add_quad_face(static_faces,
                                  global_to_local_idx_map[p0_global],
                                  global_to_local_idx_map[p1_global],
                                  global_to_local_idx_map[p2_global],
                                  global_to_local_idx_map[p3_global])
            if (num_z - 1) in [iz for iz in range(num_z)]: # Max Z face
                p0_global = get_1d_index(ix, iy, num_z - 1, num_x, num_y)
                p1_global = get_1d_index(ix + 1, iy, num_z - 1, num_x, num_y)
                p2_global = get_1d_index(ix + 1, iy + 1, num_z - 1, num_x, num_y)
                p3_global = get_1d_index(ix, iy + 1, num_z - 1, num_x, num_y)
                if all(p in boundary_1d_indices for p in [p0_global, p1_global, p2_global, p3_global]):
                    add_quad_face(static_faces,
                                  global_to_local_idx_map[p0_global],
                                  global_to_local_idx_map[p1_global],
                                  global_to_local_idx_map[p2_global],
                                  global_to_local_idx_map[p3_global])

    print(f"Detected {len(sorted_boundary_indices)} boundary vertices and {len(static_faces)} faces.")

    # --- Populate Time Steps with Animated Vertex Positions ---
    time_steps_data = []
    
    # Start with initial node coordinates
    current_nodes_positions = np.array(initial_nodes_coords)

    for t_idx, current_time in enumerate(time_points):
        # Get velocities for the current time step
        current_velocities_all_nodes = np.array(velocity_history[t_idx])

        # Calculate time step (dt)
        dt = current_time - (time_points[t_idx - 1] if t_idx > 0 else 0.0)
        # Note: If time_points are non-monotonic, dt could be negative or zero,
        # which will be caught by the monotonicity check above.

        # Update node positions using explicit Euler integration (simplistic)
        # This assumes nodes are moving based on their velocity.
        # This is a very basic physics integration; a real simulation would be more complex.
        if t_idx > 0:
            current_nodes_positions = current_nodes_positions + current_velocities_all_nodes * dt
        # For t_idx == 0, current_nodes_positions is already the initial_nodes_coords

        # Extract positions only for the identified boundary nodes
        current_frame_boundary_vertices = [
            current_nodes_positions[global_idx].tolist() for global_idx in sorted_boundary_indices
        ]

        time_steps_data.append({
            "time": float(current_time),
            "frame": t_idx,
            "vertices": current_frame_boundary_vertices,
        })

    # --- Construct Final JSON Structure ---
    output_data = {
        "mesh_name": "FluidSurface",
        "static_faces": static_faces,
        "time_steps": time_steps_data
    }

    # 4. Save JSON Output
    print(f"Saving mesh data to {output_mesh_json_path}")
    with open(output_mesh_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print("Mesh data JSON created successfully!")

# --- Main execution ---
if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to read the existing navier_stokes_results.json
    navier_stokes_file = os.path.join(current_script_dir, "../data/testing-input-output/navier_stokes_results.json")
    
    # Output path for the fluid_mesh_data.json
    output_mesh_json = os.path.join(current_script_dir, "../data/testing-input-output/fluid_mesh_data.json")

    # Call the function to generate the mesh data JSON
    try:
        generate_fluid_mesh_data_json(navier_stokes_file, output_mesh_json)
    except Exception as e:
        print(f"An error occurred during execution: {e}")



