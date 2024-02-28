import ntcore
import time
# basically copied from https://docs.wpilib.org/en/stable/docs/software/networktables/networktables-intro.html
# this has been successfully tested between local process <-> local process
# as well as local process <-> RoboRIO
# it is yet to be tested between the Orange Pi <-> RoboRIO


class NetworkTablesController:
    def __init__(self) -> None:
        self.inst = ntcore.NetworkTableInstance.getDefault()
        self.table = self.inst.getTable("datatable")
        self.distance_pub = self.table.getDoubleTopic("distance").publish()
        self.angle_pub = self.table.getDoubleTopic("angle").publish()
        self.inst.startClient4("example client")
        self.inst.setServer("localhost")
        self.inst.setServerTeam(7476, 0)
        self.inst.startDSClient()

    def send_data(self, angle, distance):
        self.angle_pub.set(angle)
        self.distance_pub.set(distance)
        print(f"Angle: {angle}, Distance: {distance}")

if __name__ == "__main__":
    pass
