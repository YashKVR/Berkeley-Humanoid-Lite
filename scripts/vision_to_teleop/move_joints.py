from move_actuator_util import move_actuator

angle_list = [[(2, 0.5)], [(2, 1.0)], [(2, 1.5)]]

for i in range(len(angle_list)):
    move_actuator(angle_list[i], len(angle_list), i)