import os
import numpy as np
import math
from scipy.interpolate import splprep, splev

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
from qvl.system import QLabsSystem
from qvl.traffic_cone import QLabsTrafficCone
import pal.resources.rtmodels as rtmodels

def generate_straight_track(track_width=4.8, spacing=2.0, track_length=60):
    # Simple straight line control points
    ctrl_x = [0, track_length]
    ctrl_y = [0, 0]
    
    # Create points along the straight line
    num_points = int(track_length / spacing)
    x_fine = np.linspace(0, track_length, num_points)
    y_fine = np.zeros(num_points)

    # Generate parallel cone lines
    yellow_cones = []
    blue_cones = []
    
    for i in range(len(x_fine)):
        # For straight track, offset is simply in Y direction
        yellow_cones.append([
            x_fine[i],
            track_width / 2,
            0
        ])
        
        blue_cones.append([
            x_fine[i],
            -track_width / 2,
            0
        ])

    start_point = [x_fine[0], y_fine[0], 0]
    end_point = [x_fine[-1], y_fine[-1], 0]
    
    return yellow_cones, blue_cones, start_point, end_point

def setup_with_cones_demo(
        initialPosition=None,
        initialOrientation=[0, 0, 0],
        rtModel=rtmodels.QCAR
    ):
    os.system('cls')
    qlabs = QuanserInteractiveLabs()
    
    try:
        qlabs.open("localhost")
    except:
        quit()

    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()

    yellow_positions, blue_positions, start_point, end_point = generate_straight_track()

    if initialPosition is None:
        initialPosition = [-5, 0, 0]  # Start behind the track

    hqcar = QLabsQCar(qlabs)
    hqcar.spawn_id(
        actorNumber=0,
        location=initialPosition,
        rotation=initialOrientation,
        waitForConfirmation=True
    )

    camera = QLabsFreeCamera(qlabs)
    # Position camera to view the straight track
    camera.spawn([30, 15, 20], [-0.3, 0, 0])
    
    hqcar.possess()

    QLabsRealTime().start_real_time_model(rtModel)

    hSystem = QLabsSystem(qlabs)
    hSystem.set_title_string('QCar Straight Track')

    # Spawn yellow cones along the track
    for i, pos in enumerate(yellow_positions):
        cone = QLabsTrafficCone(qlabs)
        cone.spawn_id(
            actorNumber=100 + i,
            location=pos,
            rotation=[0, 0, math.pi],
            scale=[1, 1, 1],
            configuration=2,
            waitForConfirmation=True
        )
        cone.set_material_properties(0, [1, 1, 0], roughness=0.5, metallic=False)
        cone.set_material_properties(1, [0, 0, 0])

    # Spawn blue cones along the track
    for i, pos in enumerate(blue_positions):
        cone = QLabsTrafficCone(qlabs)
        cone.spawn_id(
            actorNumber=200 + i,
            location=pos,
            rotation=[0, 0, math.pi],
            scale=[1, 1, 1],
            configuration=2,
            waitForConfirmation=True
        )
        cone.set_material_properties(0, [0, 0, 1], roughness=1.0, metallic=False)
        cone.set_material_properties(1, [1, 1, 1])

    hqcar.possess()

    return qlabs, hqcar, start_point, end_point

if __name__ == '__main__':
    qlabs, hqcar, start_point, end_point = setup_with_cones_demo()