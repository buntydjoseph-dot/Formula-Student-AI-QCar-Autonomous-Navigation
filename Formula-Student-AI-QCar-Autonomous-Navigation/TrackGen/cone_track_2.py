# Required Libraries
import os
import numpy as np
import math
from scipy.interpolate import splprep, splev

# QLabs + QCar Libraries
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
from qvl.system import QLabsSystem
from qvl.traffic_cone import QLabsTrafficCone
import pal.resources.rtmodels as rtmodels

def generate_spline_closed_loop(track_width=5.5, spacing=1.5):
    """
    Generate left and right cone coordinates along a closed-loop spline path.
    """
    ctrl_x = [0, 12, 25, 35, 45, 35, 25, 12, 0, -12, -25, -35, -45, -35, -25, -12]
    ctrl_y = [0, 8, 18, 35, 0, -8, -18, -35, -45, -35, -18, -8, 0, 8, 18, 35]

      # Fit closed-loop spline
    tck, _ = splprep([ctrl_x, ctrl_y], s=0, per=True)

    # Uniform sampling
    num_points = int(1 / spacing * 250)
    u_fine = np.linspace(0, 1, num_points, endpoint=False)
    x_fine, y_fine = splev(u_fine, tck)

    # Yellow cones (right side)
    yellow_cones = []
    for i in range(len(x_fine)):
        dx = x_fine[(i + 1) % len(x_fine)] - x_fine[i - 1]
        dy = y_fine[(i + 1) % len(y_fine)] - y_fine[i - 1]
        norm = np.hypot(dx, dy)
        if norm > 0:
            dx /= norm
            dy /= norm
        offset_x = -dy
        offset_y = dx
        yellow_cones.append([
            x_fine[i] + (track_width / 2) * offset_x,
            y_fine[i] + (track_width / 2) * offset_y,
            0
        ])

    # Blue cones (left side)
    blue_cones = []
    for i in range(len(x_fine)):
        dx = x_fine[(i + 1) % len(x_fine)] - x_fine[i - 1]
        dy = y_fine[(i + 1) % len(y_fine)] - y_fine[i - 1]
        norm = np.hypot(dx, dy)
        if norm > 0:
            dx /= norm
            dy /= norm
        offset_x = -dy
        offset_y = dx
        blue_cones.append([
            x_fine[i] - (track_width / 2) * offset_x,
            y_fine[i] - (track_width / 2) * offset_y,
            0
        ])

    return yellow_cones, blue_cones

def spawn_additional_blue_cones(qlabs):
    """
    Spawn additional blue cones at specified positions
    """
    additional_blue_positions = [
        [1.74, -2.355, 0],
        [-1.281, -2.639, 0],
        [-4.218, -1.545, 0]
    ]
    
    print("Spawning additional blue cones...")
    for i, pos in enumerate(additional_blue_positions):
        cone = QLabsTrafficCone(qlabs)
        cone.spawn_id(
            actorNumber=400 + i,  # Using 400+ to avoid conflicts with existing cones
            location=pos,
            rotation=[0, 0, math.pi],
            scale=[1, 1, 1],
            configuration=2,
            waitForConfirmation=True
        )
        cone.set_material_properties(0, [0, 0, 1], roughness=1.0, metallic=False)  # blue
        cone.set_material_properties(1, [1, 1, 1])

def setup_with_cones_demo(
        initialPosition=[12.862, 8.715, -0],
        initialOrientation=[0, 0, 180],
        rtModel=rtmodels.QCAR
    ):
    """
    Spawns a closed-loop racing track using yellow and blue cones and the QCar.
    """
    os.system('cls')
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    try:
        qlabs.open("localhost")
        print("Connected to QLabs")
    except:
        print("Unable to connect to QLabs")
        quit()

    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()

    # Spawn QCar
    hqcar = QLabsQCar(qlabs)
    hqcar.spawn_id(
        actorNumber=0,
        location=initialPosition,
        rotation=initialOrientation,
        waitForConfirmation=True
    )

    # Top-down camera
    camera = QLabsFreeCamera(qlabs)
    camera.spawn([8.484, 1.973, 12.209], [-0, 0.748, 0.792])
    hqcar.possess()

    QLabsRealTime().start_real_time_model(rtModel)

    hSystem = QLabsSystem(qlabs)
    hSystem.set_title_string('Traffic Cones Detection with YOLOv8 - Closed Loop Track')

    print("Generating closed loop cone coordinates...")
    yellow_positions, blue_positions = generate_spline_closed_loop()

    # Spawn yellow cones
    print("Spawning yellow cones...")
    for i, pos in enumerate(yellow_positions):
        cone = QLabsTrafficCone(qlabs)
        cone.spawn_id(
            actorNumber=300 + i,
            location=pos,
            rotation=[0, 0, math.pi],
            scale=[1, 1, 1],
            configuration=2,
            waitForConfirmation=True
        )
        cone.set_material_properties(0, [1, 1, 0], roughness=0.5, metallic=False)  # yellow
        cone.set_material_properties(1, [0, 0, 0])

    # Spawn blue cones
    print("Spawning blue cones...")
    for i, pos in enumerate(blue_positions):
        cone = QLabsTrafficCone(qlabs)
        cone.spawn_id(
            actorNumber=500 + i,
            location=pos,
            rotation=[0, 0, math.pi],
            scale=[1, 1, 1],
            configuration=2,
            waitForConfirmation=True
        )
        cone.set_material_properties(0, [0, 0, 1], roughness=1.0, metallic=False)  # blue
        cone.set_material_properties(1, [1, 1, 1])

    # Spawn additional blue cones at specified positions
    spawn_additional_blue_cones(qlabs)

    hqcar.possess()
    print("✅ Track setup complete: Closed-loop racing circuit with additional blue cones ready.")

    return qlabs, hqcar

if __name__ == '__main__':
    print("🚗 Starting QCar Closed Loop Track Setup...")
    print("=" * 50)
    qlabs, hqcar = setup_with_cones_demo()