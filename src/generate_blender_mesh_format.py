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
        return

    time_points = np.array(navier_stokes_data['time_points'])
    
    # IMPORTANT: The provided navier_stokes_results.json structure shows 'nodes_coords'
    # and 'velocity_history' at the top level, but 'nodes_coords' is under 'mesh_info'
    # and is static. 'velocity_history' is per time step.
    # To animate the mesh, we need per-time-step node coordinates.
    # Since your current navier_stokes_results.json implies static nodes_coords,
    # we'll use the 'velocity_history' to *derive* animated node positions.
    # This is a simplification; a real simulation would likely output animated
    # node positions directly.

    initial_nodes_coords = np.array(navier_stokes_data['mesh_info']['nodes_coords'])
    grid_shape = navier_stokes_data['mesh_info']['grid_shape'] # [Z, Y, X]
    velocity_history = navier_stokes_data['velocity_history'] # Per time step, per node

    num_z, num_y, num_x = grid_shape
    
    # --- Identify Surface Nodes and Create Static Faces ---
    def get_1d_index(ix, iy, iz, nx, ny):
        # Index = z_i * (num_y * num_x) + y_i * num_x + x_i
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
                    if idx_1d < len(initial_nodes_coords):
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
        dt = time_points[t_idx] - (time_points[t_idx - 1] if t_idx > 0 else 0.0)

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
    generate_fluid_mesh_data_json(navier_stokes_file, output_mesh_json)
