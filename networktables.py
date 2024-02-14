import ntcore
import time

# basically copied from https://docs.wpilib.org/en/stable/docs/software/networktables/networktables-intro.html
# this has been successfully tested between local process <-> local process
# as well as local process <-> RoboRIO
# it is yet to be tested between the Orange Pi <-> RoboRIO

if __name__ == "__main__":
    inst = ntcore.NetworkTableInstance.getDefault()
    table = inst.getTable("datatable")
    xSub = table.getDoubleTopic("x").publish()
    ySub = table.getDoubleTopic("y").publish()
    inst.startClient4("example client")
    inst.setServer("localhost")
    inst.setServerTeam(7476, 0)
    inst.startDSClient() # recommended if running on DS computer; this gets the robot IP from the DS

    counter = 0.0
    while True:
        time.sleep(0.5)

        x = xSub.set(counter)
        y = ySub.set(counter)
        counter += 1
        print(f"X: {x} Y: {y}")
