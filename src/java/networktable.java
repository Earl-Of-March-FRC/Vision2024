package frc.robot.Subsystems;

import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableEntry;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.wpilibj2.command.SubsystemBase;

public class VisionSubsystem extends SubsystemBase {
    private final NetworkTableInstance inst;

    private final NetworkTable table;
    private final NetworkTableEntry distanceEntry;
    private final NetworkTableEntry angleEntry;

    public VisionSubsystem() {
        inst = NetworkTableInstance.getDefault();
        inst.startServer(); // Start the server on the Java side
        inst.setServerTeam(9127, 0); // Set the team number (could also be 7476)

        table = inst.getTable("vision"); // Updated table name
        distanceEntry = table.getEntry("distance");
        angleEntry = table.getEntry("angle");
    }

    @Override
    public void periodic() {
        System.out.println(getDistance() + ", " + getAngle());
    }

    public double getDistance() {
        // Get distance from NetworkTable
        return distanceEntry.getDouble(-1.0); // Default value is -1.0 if entry is not found
    }

    public double getAngle() {
        // Get angle from NetworkTable
        return angleEntry.getDouble(-1.0); // Default value is -1.0 if entry is not found
    }

    public boolean hasDistance() {
        return (getDistance() < 0 ? false : true);
    }

    public boolean hasAngle() {
        return (getAngle() < 0 ? false : true);
    }
}
