package frc.robot.Subsystems;

import edu.wpi.first.math.geometry.Pose3d;
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
      inst.startClient3("example client");
      inst.setServer("localhost");
      inst.setServerTeam(9127, 0);

      table = inst.getTable("datatable");
      distanceEntry = table.getEntry("distance");
      angleEntry = table.getEntry("angle");
  }

  @Override
  public void periodic() {
    System.out.println(getDistance() + ", " + getAngle());
  }

  public double getDistance() {
      // Get distance from NetworkTable
      return distanceEntry.getDouble(0.0); // Default value is 0.0 if entry is not found
  }

  public double getAngle() {
      // Get angle from NetworkTable
      return angleEntry.getDouble(0.0); // Default value is 0.0 if entry is not found
  }
}
