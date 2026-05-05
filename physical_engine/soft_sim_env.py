import genesis as gs
import numpy as np
import trimesh
from pathlib import Path

class SoftSimEnv:
    def __init__(self):
        self.scene = None
        self.solver = None
        self.sensor_cube = None
        self.elastoplastic_obj = None

    def setup_simulation(self, elastoplastic_shape="sphere", mesh_file_path=None):
        gs.init(seed=42, precision='32', logging_level='warning', theme='light', debug=False)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=2e-3, substeps=20, gravity=(0, 0, -5.8)),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.6, -0.6, -0.5), upper_bound=(0.6, 0.6, 1.0),
                grid_density=128, particle_size=0.007),
            vis_options=gs.options.VisOptions(
                visualize_mpm_boundary=False, background_color=(0.53, 0.81, 0.98),
                lights=[
                    {
                        "type": "directional",
                        "dir": (-1, -1, -1),
                        "color": (0.8, 0.8, 0.8),
                        "intensity": 5.0
                    },
                    {
                        "type": "directional",
                        "dir": (0, -1, -0.2),
                        "color": (0.8, 0.8, 0.8),
                        "intensity": 5.0
                    },
                    {
                        "type": "directional",
                        "dir": (-1, 0, -0.2),
                        "color": (0.8, 0.8, 0.8),
                        "intensity": 5.0
                    }
                ]
            ),
            viewer_options=gs.options.ViewerOptions(camera_fov=35),
            show_viewer=True,
        )

        self._setup_materials()
        self._setup_entities(elastoplastic_shape, mesh_file_path, scale=1.0)
        self.scene.build()
        self.solver = self._get_solver()

        return self.scene, self.solver, self.sensor_cube, self.elastoplastic_obj

    def _setup_materials(self):
        self.elastic_mat = gs.materials.MPM.Elastic(E=1e6, nu=0.47, rho=5150.0, sampler='regular', model="neohooken")
        self.elasto_plastic_mat = gs.materials.MPM.ElastoPlastic(
            E=7e4+5000, nu=0.2, rho=1000.0, von_mises_yield_stress=30000.0,
            use_von_mises=True, sampler='random')

    def _setup_entities(self, elastoplastic_shape="sphere", mesh_file_path=None, scale=1.0):
        self.scene.add_entity(
            morph=gs.morphs.Plane(),
            material=gs.materials.Rigid(
                rho=200.0, friction=4.0, needs_coup=True,
                coup_friction=0.8, coup_softness=0.002, coup_restitution=0.0
            )
        )

        if elastoplastic_shape == "mesh" and mesh_file_path:
            self.elastoplastic_obj = self._create_mesh_shape(mesh_file_path, scale)
        else:
            self.elastoplastic_obj = self._create_elastoplastic_shape(
                shape_type=elastoplastic_shape,
                pos=(0, 0, 0)
            )

        cube_h = 0.1
        cube_center_z = 0.3
        self.sensor_cube = self.scene.add_entity(
            material=self.elastic_mat,
            morph=gs.morphs.Box(pos=(0, 0, cube_center_z), size=(0.4, 0.4, cube_h)),
            surface=gs.surfaces.Default(color=(0.2, 0.6, 0.8), vis_mode='particle'),
        )

    def _create_elastoplastic_shape(self, shape_type="sphere", pos=(0, 0, 0)):
        x, y, z = pos

        if shape_type == "sphere":
            radius = 0.08
            adjusted_z = radius
            morph = gs.morphs.Sphere(pos=(x, y, adjusted_z), radius=radius)
            color = (1.0, 0.2, 0.0)

        elif shape_type == "box":
            size_x, size_y, size_z = 0.15, 0.15, 0.15
            adjusted_z = size_z / 2
            morph = gs.morphs.Box(pos=(x, y, adjusted_z), size=(size_x, size_y, size_z))
            color = (0.2, 1.0, 0.2)

        elif shape_type == "cylinder":
            height = 0.15
            radius = 0.07
            adjusted_z = height / 2
            morph = gs.morphs.Cylinder(pos=(x, y, adjusted_z), height=height, radius=radius)
            color = (0.8, 0.8, 0.2)

        else:
            raise ValueError(f"Unsupported shape types: {shape_type}")

        return self.scene.add_entity(
            material=self.elasto_plastic_mat,
            morph=morph,
            surface=gs.surfaces.Default(color=color, vis_mode='particle'),
        )

    def _create_mesh_shape(self, mesh_file_path, scale=1.0):
        if not Path(mesh_file_path).exists():
            raise FileNotFoundError(f"The mesh file does not exist.: {mesh_file_path}")

        print(f"Load mesh file: {mesh_file_path}")
        mesh = trimesh.load_mesh(mesh_file_path)

        if not mesh.is_watertight:
            print("Warning: The grid is not watertight. Attempting to repair it...")
            mesh.fill_holes()

        volume = mesh.volume
        bounds = mesh.bounds
        center = mesh.centroid
        size = bounds[1] - bounds[0]

        print(f"Mesh Information: volume={volume:.6f}, center={center}, size={size}")

        morph = gs.morphs.Mesh(
            file=mesh_file_path,
            scale=scale,
            pos=(0, 0, -bounds[0][2]),
            decimate=False,
            convexify=False,
        )
        entity = self.scene.add_entity(
            material=self.elasto_plastic_mat,
            morph=morph,
            surface=gs.surfaces.Default(color=(0.9, 0.4, 0.1), vis_mode='particle'),
        )
        return entity

    def _get_solver(self, solver_type="mpm"):
        if hasattr(self.scene, f"{solver_type}_solver"):
            return getattr(self.scene, f"{solver_type}_solver")
        elif hasattr(self.scene, "solvers") and solver_type in self.scene.solvers:
            return self.scene.solvers[solver_type]
        else:
            raise AttributeError(f"[ERROR] {solver_type}_solver not found in the Scene.")


    def initialize_cube_particles(self):
        initial_particles = self.sensor_cube.get_particles()
        if hasattr(initial_particles, 'cpu'):
            initial_particles = initial_particles.cpu().numpy()
        initial_particles = initial_particles.reshape(-1, 3)

        free_mask = np.ones(len(initial_particles), dtype=np.int32)
        self.sensor_cube.set_free(free_mask)

        return initial_particles