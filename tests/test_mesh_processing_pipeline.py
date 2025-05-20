import unittest
import os
import json
import numpy as np
import alembic  # Ensure you have the 'alembic' Python package installed
from jsonschema import validate


class TestAlembicFluidSurfaceProcessing(unittest.TestCase):

    def setUp(self):
        """Define paths for input metadata and Alembic output"""
        self.metadata_path = "data/testing-input-output/fluid_mesh_metadata.json"
        self.output_abc_path = "data/testing-input-output/fluid_mesh.abc"

    def test_json_schema(self):
        """Ensure fluid surface metadata follows expected schema"""
        with open(self.metadata_path) as f:
            mesh_data = json.load(f)

        schema = {
            "type": "object",
            "properties": {
                "mesh_info": {"type": "object"},
                "global_parameters": {"type": "object"},
                "vertices": {"type": "array"}
            },
            "required": ["mesh_info", "global_parameters", "vertices"]
        }
        validate(instance=mesh_data, schema=schema)

    def test_mesh_vertex_structure(self):
        """Ensure fluid surface contains expected vertex attributes"""
        with open(self.metadata_path) as f:
            mesh_data = json.load(f)

        assert len(mesh_data["vertices"]) > 0, "Mesh vertices missing!"
        assert all("x" in v and "y" in v and "z" in v for v in mesh_data["vertices"]), "Vertex components missing!"

    def test_physical_consistency(self):
        """Ensure fluid surface properties remain physically realistic"""
        with open(self.metadata_path) as f:
            mesh_data = json.load(f)

        assert mesh_data["global_parameters"]["surface_tension"]["value"] > 0, "Surface tension should be positive!"
        assert mesh_data["global_parameters"]["density"]["value"] == 1000, "Density mismatch detected!"

    def test_binary_mesh_integrity(self):
        """Ensure structured binary data is correctly formatted in .npy file"""
        np_data = np.load("data/testing-input-output/fluid_mesh.npy")
        assert np_data.shape[0] > 0, "Mesh data structure invalid!"
        assert "vertex_x" in np_data.dtype.names and "vertex_y" in np_data.dtype.names and "vertex_z" in np_data.dtype.names, "Vertex fields missing!"

    ### ALEMBIC FILE VALIDATION TESTS ###
    
    def test_alembic_file_exists(self):
        """Ensure the Alembic (.abc) output file is created."""
        self.assertTrue(os.path.exists(self.output_abc_path), f"Alembic output file not found at {self.output_abc_path}!")

    def test_alembic_file_is_valid(self):
        """Check if the generated Alembic file can be opened and is valid."""
        if os.path.exists(self.output_abc_path):
            try:
                archive = alembic.Abc.IArchive(self.output_abc_path)
                self.assertTrue(archive.isValid(), "The Alembic file is not valid.")
                archive.close()
            except Exception as e:
                self.fail(f"Failed to open or validate Alembic file: {e}")
        else:
            self.fail("Alembic file does not exist, cannot test validity.")

    def test_alembic_contains_geometry(self):
        """Check if the Alembic file contains at least one geometric object."""
        if os.path.exists(self.output_abc_path):
            archive = alembic.Abc.IArchive(self.output_abc_path)
            top_node = archive.getTop()
            found_geometry = False
            for child in top_node.children:
                if alembic.AbcGeom.IXform.matches(child.getHeader()) or \
                   alembic.AbcGeom.IPolyMesh.matches(child.getHeader()) or \
                   alembic.AbcGeom.ISubD.matches(child.getHeader()):
                    found_geometry = True
                    break
            self.assertTrue(found_geometry, "Alembic file does not contain any geometry.")
            archive.close()
        else:
            self.fail("Alembic file does not exist, cannot test geometry.")

    def test_alembic_vertex_count(self):
        """Check if the mesh in the Alembic file has a reasonable number of vertices."""
        if os.path.exists(self.output_abc_path):
            archive = alembic.Abc.IArchive(self.output_abc_path)
            top_node = archive.getTop()
            for child in top_node.children:
                if alembic.AbcGeom.IPolyMesh.matches(child.getHeader()):
                    poly_mesh = alembic.AbcGeom.IPolyMesh(child, alembic.Abc.kWrapExisting)
                    points_prop = poly_mesh.getSchema().getPositionsProperty()
                    sample = points_prop.getValue()
                    self.assertGreater(len(sample), 0, "The mesh in the Alembic file has no vertices.")
                    break
            archive.close()
        else:
            self.fail("Alembic file does not exist, cannot test vertex count.")


if __name__ == "__main__":
    unittest.main()



