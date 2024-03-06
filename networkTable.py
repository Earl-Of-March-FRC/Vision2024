import ntcore
import logging
logging.basicConfig(level=logging.DEBUG)


class NetworkTable:
    def __init__(self, *, instance: ntcore.NetworkTableInstance | None = None):
        """
        READ THIS FOR MORE INFO: https://docs.wpilib.org/en/stable/docs/software/networktables/networktables-networking.html

        The robot is the server, therefore this script is the client, and so this table instance should be set up as such
        """

        self._inst = instance or ntcore.NetworkTableInstance.getDefault()
        self._table = self._inst.getTable("vision")
        self._distance = self._table.getDoubleTopic("distance").publish()
        self._angle = self._table.getDoubleTopic("angle").publish()

        self._inst.startClient4("vision client")
        # self._inst.setServerTeam(7476, 0)
        self._inst.setServer("10.74.76.227", port=ntcore.NetworkTableInstance.kDefaultPort4)

    def send_data(self, distance: float, angle: float) -> None:
        self._distance.set(distance)
        self._angle.set(angle)
        logging.debug("Angle: %.2f, Distance: %.2f", angle, distance)

    def close(self):
        self._inst.stopClient()

    @property
    def instance(self) -> ntcore.NetworkTableInstance:
        return self._inst
